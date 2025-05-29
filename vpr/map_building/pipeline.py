# vpr/training/map_building/pipeline.py

import os
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import torch
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models.vlm.vlm_utils import load_vlm_model
from ..models.vlad.vlad_trainer import VLADEncoder
from ..models.octree.octree_builder import OctreeBuilder
from .feature_extraction import FeatureExtractor
from .descriptor_generation import DescriptorGenerator
from .foursquare_integration import FoursquareEnricher
from .map_optimizer import MapOptimizer
from ..utils.logging import setup_logger
from ..utils.memory_manager import MemoryManager
from ..utils.checkpointing import CheckpointManager

logger = setup_logger(__name__)


@dataclass
class MapBuildingConfig:
    """Configuration for map building pipeline"""
    # Octree parameters
    octree_max_depth: int = 12
    octree_min_node_size: float = 1.0  # meters
    octree_max_node_size: float = 1000.0  # meters
    
    # Feature extraction
    feature_resolutions: List[Tuple[int, int]] = None
    keyframe_interval: int = 10
    batch_size: int = 32
    
    # VLAD parameters
    vlad_clusters: int = 64
    vlad_dim: int = 4096
    
    # Foursquare integration
    foursquare_enabled: bool = True
    foursquare_radius: int = 50  # meters
    foursquare_categories: List[str] = None
    
    # Processing
    num_workers: int = 8
    device: str = "cuda"
    mixed_precision: bool = True
    checkpoint_interval: int = 100
    
    def __post_init__(self):
        if self.feature_resolutions is None:
            self.feature_resolutions = [(640, 480), (320, 240), (160, 120)]
        if self.foursquare_categories is None:
            self.foursquare_categories = ["landmark", "transportation", "food", "shops"]


class MapBuildingPipeline:
    """Main pipeline for building VPR maps"""
    
    def __init__(self, config: MapBuildingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.feature_extractor = None
        self.descriptor_generator = None
        self.octree_builder = None
        self.foursquare_enricher = None
        self.map_optimizer = None
        
        # Memory and checkpoint managers
        self.memory_manager = MemoryManager(max_memory_gb=8)
        self.checkpoint_manager = CheckpointManager()
        
        # Statistics
        self.stats = {
            "images_processed": 0,
            "keyframes_extracted": 0,
            "descriptors_generated": 0,
            "octree_nodes_created": 0,
            "pois_added": 0
        }
    
    def initialize(self, vlm_model_path: str, vlad_model_path: Optional[str] = None,
                   foursquare_api_key: Optional[str] = None):
        """Initialize all components of the pipeline"""
        logger.info("Initializing map building pipeline...")
        
        # Load VLM model
        vlm_model = load_vlm_model(vlm_model_path, self.device, quantized=True)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            vlm_model=vlm_model,
            resolutions=self.config.feature_resolutions,
            device=self.device
        )
        
        # Initialize VLAD encoder
        if vlad_model_path and os.path.exists(vlad_model_path):
            logger.info(f"Loading VLAD model from {vlad_model_path}")
            self.descriptor_generator = DescriptorGenerator.load(vlad_model_path)
        else:
            logger.info("Initializing new VLAD encoder")
            self.descriptor_generator = DescriptorGenerator(
                feature_dim=self.feature_extractor.get_feature_dim(),
                num_clusters=self.config.vlad_clusters,
                descriptor_dim=self.config.vlad_dim
            )
        
        # Initialize octree builder
        self.octree_builder = OctreeBuilder(
            max_depth=self.config.octree_max_depth,
            min_node_size=self.config.octree_min_node_size,
            max_node_size=self.config.octree_max_node_size
        )
        
        # Initialize Foursquare enricher if enabled
        if self.config.foursquare_enabled and foursquare_api_key:
            self.foursquare_enricher = FoursquareEnricher(
                api_key=foursquare_api_key,
                search_radius=self.config.foursquare_radius,
                categories=self.config.foursquare_categories
            )
        
        # Initialize map optimizer
        self.map_optimizer = MapOptimizer()
        
        logger.info("Pipeline initialization complete")
    
    def build_map(self, dataset_path: str, output_path: str, 
                  resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Build a complete map from a dataset
        
        Args:
            dataset_path: Path to the dataset
            output_path: Path to save the map
            resume_from: Optional checkpoint to resume from
            
        Returns:
            Dictionary with map statistics and metadata
        """
        logger.info(f"Building map from dataset: {dataset_path}")
        
        # Load dataset metadata
        metadata = self._load_dataset_metadata(dataset_path)
        
        # Resume from checkpoint if provided
        start_idx = 0
        if resume_from:
            checkpoint = self.checkpoint_manager.load(resume_from)
            self.octree_builder.load_state(checkpoint['octree_state'])
            self.stats = checkpoint['stats']
            start_idx = checkpoint['last_processed_idx'] + 1
            logger.info(f"Resuming from checkpoint at index {start_idx}")
        
        # Process images in batches
        image_paths = metadata['images'][start_idx:]
        poses = metadata['poses'][start_idx:]
        
        # Phase 1: Extract features and build initial map
        logger.info("Phase 1: Feature extraction and initial map building")
        self._process_images(image_paths, poses, start_idx)
        
        # Phase 2: Train VLAD if needed
        if not self.descriptor_generator.is_trained():
            logger.info("Phase 2: Training VLAD encoder")
            self._train_vlad()
        
        # Phase 3: Generate descriptors and populate octree
        logger.info("Phase 3: Generating descriptors and populating octree")
        self._generate_descriptors()
        
        # Phase 4: Enrich with Foursquare data
        if self.foursquare_enricher:
            logger.info("Phase 4: Enriching map with Foursquare POIs")
            self._enrich_with_foursquare()
        
        # Phase 5: Optimize map
        logger.info("Phase 5: Optimizing map structure")
        self._optimize_map()
        
        # Save final map
        logger.info(f"Saving map to {output_path}")
        map_data = self._save_map(output_path)
        
        return {
            "map_path": output_path,
            "statistics": self.stats,
            "metadata": map_data
        }
    
    def _process_images(self, image_paths: List[str], poses: List[np.ndarray], 
                        start_idx: int = 0):
        """Process images to extract features and build initial octree structure"""
        
        # Create batches
        batches = []
        for i in range(0, len(image_paths), self.config.batch_size):
            batch_images = image_paths[i:i + self.config.batch_size]
            batch_poses = poses[i:i + self.config.batch_size]
            batch_indices = list(range(start_idx + i, start_idx + i + len(batch_images)))
            batches.append((batch_images, batch_poses, batch_indices))
        
        # Process batches
        with tqdm(total=len(image_paths), desc="Processing images") as pbar:
            for batch_images, batch_poses, batch_indices in batches:
                # Extract features
                features = self.feature_extractor.extract_batch(batch_images)
                
                # Process each image in batch
                for idx, (image_path, pose, image_features) in enumerate(
                    zip(batch_images, batch_poses, features)):
                    
                    # Check if this is a keyframe
                    if self._is_keyframe(batch_indices[idx]):
                        # Add to octree
                        position = pose[:3, 3]  # Extract translation
                        node = self.octree_builder.insert_point(position)
                        
                        # Store features temporarily
                        node.temp_features = image_features
                        node.image_path = image_path
                        node.pose = pose
                        
                        self.stats["keyframes_extracted"] += 1
                        self.stats["octree_nodes_created"] += 1
                    
                    self.stats["images_processed"] += 1
                    pbar.update(1)
                
                # Checkpoint periodically
                if batch_indices[-1] % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(batch_indices[-1])
                
                # Memory management
                self.memory_manager.check_and_cleanup()
    
    def _train_vlad(self):
        """Train VLAD encoder on extracted features"""
        
        # Collect all features from octree nodes
        all_features = []
        nodes = self.octree_builder.get_all_leaf_nodes()
        
        logger.info(f"Collecting features from {len(nodes)} nodes for VLAD training")
        
        for node in tqdm(nodes, desc="Collecting features"):
            if hasattr(node, 'temp_features'):
                all_features.append(node.temp_features)
        
        # Convert to numpy array
        all_features = np.vstack(all_features)
        
        # Train VLAD
        logger.info(f"Training VLAD on {len(all_features)} feature vectors")
        self.descriptor_generator.train(all_features)
        
        # Save trained VLAD model
        vlad_path = "models/vlad_encoder/trained_vlad.pth"
        os.makedirs(os.path.dirname(vlad_path), exist_ok=True)
        self.descriptor_generator.save(vlad_path)
        logger.info(f"Saved trained VLAD model to {vlad_path}")
    
    def _generate_descriptors(self):
        """Generate VLAD descriptors for all nodes"""
        
        nodes = self.octree_builder.get_all_leaf_nodes()
        
        with tqdm(total=len(nodes), desc="Generating descriptors") as pbar:
            for node in nodes:
                if hasattr(node, 'temp_features'):
                    # Generate VLAD descriptor
                    vlad_descriptor = self.descriptor_generator.encode(node.temp_features)
                    
                    # Store in node
                    node.vlad_descriptor = vlad_descriptor
                    
                    # Clean up temporary features to save memory
                    delattr(node, 'temp_features')
                    
                    self.stats["descriptors_generated"] += 1
                    pbar.update(1)
    
    def _enrich_with_foursquare(self):
        """Enrich map with Foursquare POI data"""
        
        nodes = self.octree_builder.get_all_leaf_nodes()
        
        # Process in parallel for efficiency
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            
            for node in nodes:
                if hasattr(node, 'pose'):
                    position = node.pose[:3, 3]
                    lat, lon = self._position_to_latlon(position)
                    
                    future = executor.submit(
                        self.foursquare_enricher.enrich_location,
                        lat, lon, node
                    )
                    futures.append((future, node))
            
            # Process results
            with tqdm(total=len(futures), desc="Enriching with Foursquare") as pbar:
                for future, node in futures:
                    try:
                        pois = future.result()
                        if pois:
                            node.pois = pois
                            self.stats["pois_added"] += len(pois)
                    except Exception as e:
                        logger.warning(f"Failed to enrich node: {e}")
                    
                    pbar.update(1)
    
    def _optimize_map(self):
        """Optimize the map structure"""
        
        logger.info("Optimizing octree structure...")
        
        # Optimize octree
        self.octree_builder.optimize()
        
        # Build navigation graph
        logger.info("Building navigation graph...")
        self.map_optimizer.build_navigation_graph(self.octree_builder)
        
        # Compress descriptors
        logger.info("Compressing descriptors...")
        self.map_optimizer.compress_descriptors(self.octree_builder)
        
        # Compute statistics
        map_stats = self.map_optimizer.compute_statistics(self.octree_builder)
        self.stats.update(map_stats)
    
    def _save_map(self, output_path: str) -> Dict[str, Any]:
        """Save the completed map"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare map data
        map_data = {
            "octree": self.octree_builder.serialize(),
            "metadata": {
                "creation_date": str(np.datetime64('now')),
                "statistics": self.stats,
                "config": self.config.__dict__,
                "vlad_model_path": "models/vlad_encoder/trained_vlad.pth"
            }
        }
        
        # Save map
        torch.save(map_data, output_path)
        
        # Save human-readable metadata
        metadata_path = output_path.replace('.octree', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(map_data["metadata"], f, indent=2)
        
        logger.info(f"Map saved successfully to {output_path}")
        return map_data["metadata"]
    
    def _load_dataset_metadata(self, dataset_path: str) -> Dict[str, Any]:
        """Load dataset metadata"""
        
        metadata_path = os.path.join(dataset_path, "metadata.json")
        if not os.path.exists(metadata_path):
            # Generate metadata by scanning directory
            metadata = self._generate_dataset_metadata(dataset_path)
            
            # Save for future use
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        else:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return metadata
    
    def _generate_dataset_metadata(self, dataset_path: str) -> Dict[str, Any]:
        """Generate metadata by scanning dataset directory"""
        
        logger.info("Generating dataset metadata...")
        
        images_dir = os.path.join(dataset_path, "images")
        poses_dir = os.path.join(dataset_path, "poses")
        
        # Get sorted list of images
        image_files = sorted([
            f for f in os.listdir(images_dir) 
            if f.endswith(('.jpg', '.png'))
        ])
        
        # Load corresponding poses
        images = []
        poses = []
        
        for image_file in tqdm(image_files, desc="Loading dataset"):
            image_path = os.path.join(images_dir, image_file)
            pose_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
            pose_path = os.path.join(poses_dir, pose_file)
            
            if os.path.exists(pose_path):
                pose = np.loadtxt(pose_path).reshape(4, 4)
                images.append(image_path)
                poses.append(pose.tolist())
        
        metadata = {
            "dataset_path": dataset_path,
            "num_images": len(images),
            "images": images,
            "poses": poses
        }
        
        return metadata
    
    def _is_keyframe(self, index: int) -> bool:
        """Determine if an image should be treated as a keyframe"""
        return index % self.config.keyframe_interval == 0
    
    def _position_to_latlon(self, position: np.ndarray) -> Tuple[float, float]:
        """Convert 3D position to latitude/longitude"""
        # This is a placeholder - implement based on your coordinate system
        # For now, assume position is in a local metric coordinate system
        # centered at a reference lat/lon
        
        ref_lat = 37.7749  # San Francisco
        ref_lon = -122.4194
        
        # Simple linear approximation (not accurate for large distances)
        meters_per_degree_lat = 111319.9
        meters_per_degree_lon = meters_per_degree_lat * np.cos(np.radians(ref_lat))
        
        lat = ref_lat + position[1] / meters_per_degree_lat
        lon = ref_lon + position[0] / meters_per_degree_lon
        
        return lat, lon
    
    def _save_checkpoint(self, last_processed_idx: int):
        """Save checkpoint for resuming"""
        
        checkpoint = {
            "last_processed_idx": last_processed_idx,
            "octree_state": self.octree_builder.get_state(),
            "stats": self.stats
        }
        
        checkpoint_path = f"checkpoints/map_building_checkpoint_{last_processed_idx}.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        self.checkpoint_manager.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint at index {last_processed_idx}")


def main():
    """Main function for standalone execution"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Build VPR map from dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--vlm_model", type=str, required=True, help="Path to VLM model")
    parser.add_argument("--vlad_model", type=str, help="Path to pre-trained VLAD model")
    parser.add_argument("--foursquare_key", type=str, help="Foursquare API key")
    parser.add_argument("--output", type=str, required=True, help="Output map path")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config object
    config = MapBuildingConfig(**config_dict.get('map_building', {}))
    
    # Create pipeline
    pipeline = MapBuildingPipeline(config)
    
    # Initialize
    pipeline.initialize(
        vlm_model_path=args.vlm_model,
        vlad_model_path=args.vlad_model,
        foursquare_api_key=args.foursquare_key
    )
    
    # Build map
    result = pipeline.build_map(
        dataset_path=args.dataset,
        output_path=args.output,
        resume_from=args.resume
    )
    
    # Print results
    print("\nMap building complete!")
    print(f"Map saved to: {result['map_path']}")
    print("\nStatistics:")
    for key, value in result['statistics'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
