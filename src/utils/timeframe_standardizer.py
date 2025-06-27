"""
Timeframe Standardization Script
Addresses the critical folder naming inconsistency found in Phase 3 testing

ISSUE: Multiple folders for same timeframes causing data fragmentation:
- 15min vs 15m (same thing)
- daily vs 1d (same thing) 
- 1hour vs 1h (same thing)

SOLUTION: Standardize to: 1m, 5m, 15m, 30m, 1h, 4h, 1d
"""
import os
import shutil
import glob
import json
from datetime import datetime
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeframeStandardizer:
    """
    Standardizes timeframe folder names and consolidates data
    """
    
    def __init__(self, base_data_path: str = "data"):
        self.base_data_path = base_data_path
        
        # Standard timeframe mapping: old_name -> new_name
        self.timeframe_mapping = {
            # Current inconsistencies found
            "15min": "15m",
            "daily": "1d", 
            "1hour": "1h",
            
            # Other potential variations to standardize
            "1min": "1m",
            "5min": "5m",
            "30min": "30m",
            "4hour": "4h",
            "1day": "1d",
            "60m": "1h",
            "240m": "4h",
            "1440m": "1d"
        }
        
        # Standard timeframes (target structure)
        self.standard_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        
        # Data types to process
        self.data_types = ["raw", "processed"]
        
        # Report for tracking changes
        self.consolidation_report = {
            "timestamp": datetime.now().isoformat(),
            "actions_taken": [],
            "files_moved": 0,
            "folders_removed": 0,
            "errors": []
        }
    
    def scan_current_structure(self) -> Dict[str, List[str]]:
        """
        Scan current data structure to identify timeframe folders
        
        Returns:
            Dictionary with data_type -> list of timeframe folders
        """
        structure = {}
        
        for data_type in self.data_types:
            data_path = os.path.join(self.base_data_path, data_type)
            structure[data_type] = []
            
            if os.path.exists(data_path):
                folders = [f for f in os.listdir(data_path) 
                          if os.path.isdir(os.path.join(data_path, f))]
                structure[data_type] = sorted(folders)
                logger.info(f"Found in {data_type}: {folders}")
            else:
                logger.warning(f"Data path not found: {data_path}")
        
        return structure
    
    def identify_duplicates(self, structure: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
        """
        Identify duplicate timeframe folders that need consolidation
        
        Returns:
            Dictionary with data_type -> list of (old_folder, new_folder) tuples
        """
        duplicates = {}
        
        for data_type, folders in structure.items():
            duplicates[data_type] = []
            
            for old_folder in folders:
                if old_folder in self.timeframe_mapping:
                    new_folder = self.timeframe_mapping[old_folder]
                    duplicates[data_type].append((old_folder, new_folder))
                    logger.info(f"Identified duplicate: {old_folder} -> {new_folder}")
        
        return duplicates
    
    def create_backup(self) -> str:
        """Create backup of current data structure"""
        backup_path = f"data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if os.path.exists(self.base_data_path):
                shutil.copytree(self.base_data_path, backup_path)
                logger.info(f"✅ Backup created: {backup_path}")
                return backup_path
            else:
                logger.warning(f"Base data path not found: {self.base_data_path}")
                return ""
        except Exception as e:
            logger.error(f"❌ Backup failed: {str(e)}")
            return ""
    
    def consolidate_timeframes(self, duplicates: Dict[str, List[Tuple[str, str]]], 
                              create_backup: bool = True) -> bool:
        """
        Consolidate duplicate timeframe folders
        
        Args:
            duplicates: Dictionary of duplicates to consolidate
            create_backup: Whether to create backup before changes
            
        Returns:
            True if successful, False otherwise
        """
        if create_backup:
            backup_path = self.create_backup()
            if not backup_path:
                logger.error("❌ Cannot proceed without backup")
                return False
            self.consolidation_report["backup_path"] = backup_path
        
        success = True
        
        for data_type, folder_pairs in duplicates.items():
            for old_folder, new_folder in folder_pairs:
                try:
                    success &= self._consolidate_single_timeframe(data_type, old_folder, new_folder)
                except Exception as e:
                    logger.error(f"❌ Error consolidating {old_folder} -> {new_folder}: {str(e)}")
                    self.consolidation_report["errors"].append(f"{old_folder} -> {new_folder}: {str(e)}")
                    success = False
        
        return success
    
    def _consolidate_single_timeframe(self, data_type: str, old_folder: str, new_folder: str) -> bool:
        """
        Consolidate a single timeframe folder
        
        Args:
            data_type: "raw" or "processed"
            old_folder: Old folder name (e.g., "15min")
            new_folder: New folder name (e.g., "15m")
            
        Returns:
            True if successful
        """
        old_path = os.path.join(self.base_data_path, data_type, old_folder)
        new_path = os.path.join(self.base_data_path, data_type, new_folder)
        
        if not os.path.exists(old_path):
            logger.warning(f"⚠️ Old path doesn't exist: {old_path}")
            return True  # Nothing to do
        
        logger.info(f"🔄 Consolidating: {old_path} -> {new_path}")
        
        # Create new folder if it doesn't exist
        os.makedirs(new_path, exist_ok=True)
        
        # Get all files in old folder
        old_files = glob.glob(os.path.join(old_path, "*"))
        
        if not old_files:
            logger.info(f"📁 Empty folder: {old_path}")
            # Remove empty old folder
            os.rmdir(old_path)
            self.consolidation_report["folders_removed"] += 1
            self.consolidation_report["actions_taken"].append(f"Removed empty folder: {old_path}")
            return True
        
        # Move/merge files
        files_moved = 0
        for old_file_path in old_files:
            filename = os.path.basename(old_file_path)
            new_file_path = os.path.join(new_path, filename)
            
            try:
                if os.path.exists(new_file_path):
                    # File exists in target - need to handle duplicate
                    if self._are_files_identical(old_file_path, new_file_path):
                        logger.info(f"🔗 Identical file, removing duplicate: {filename}")
                        os.remove(old_file_path)
                    else:
                        # Rename with timestamp to avoid conflicts
                        timestamp = datetime.fromtimestamp(os.path.getmtime(old_file_path)).strftime('%Y%m%d_%H%M%S')
                        name, ext = os.path.splitext(filename)
                        new_filename = f"{name}_{timestamp}{ext}"
                        new_file_path = os.path.join(new_path, new_filename)
                        shutil.move(old_file_path, new_file_path)
                        logger.info(f"📄 Moved with timestamp: {filename} -> {new_filename}")
                else:
                    # Simple move
                    shutil.move(old_file_path, new_file_path)
                    logger.info(f"📄 Moved: {filename}")
                
                files_moved += 1
                
            except Exception as e:
                logger.error(f"❌ Error moving {filename}: {str(e)}")
                return False
        
        # Remove old folder if empty
        try:
            if not os.listdir(old_path):  # Check if empty
                os.rmdir(old_path)
                logger.info(f"🗑️ Removed empty folder: {old_path}")
                self.consolidation_report["folders_removed"] += 1
        except Exception as e:
            logger.warning(f"⚠️ Could not remove old folder {old_path}: {str(e)}")
        
        self.consolidation_report["files_moved"] += files_moved
        self.consolidation_report["actions_taken"].append(
            f"Consolidated {old_folder} -> {new_folder}: {files_moved} files moved"
        )
        
        return True
    
    def _are_files_identical(self, file1: str, file2: str) -> bool:
        """Check if two files are identical (by size and modification time)"""
        try:
            stat1 = os.stat(file1)
            stat2 = os.stat(file2)
            
            # Compare size and modification time
            return (stat1.st_size == stat2.st_size and 
                   abs(stat1.st_mtime - stat2.st_mtime) < 2)  # 2 second tolerance
        except Exception:
            return False
    
    def validate_structure(self) -> Dict[str, List[str]]:
        """Validate final structure after consolidation"""
        logger.info("🔍 Validating final structure...")
        
        final_structure = self.scan_current_structure()
        
        issues = []
        for data_type, folders in final_structure.items():
            non_standard = [f for f in folders if f not in self.standard_timeframes]
            if non_standard:
                issues.extend(non_standard)
                logger.warning(f"⚠️ Non-standard folders in {data_type}: {non_standard}")
        
        if not issues:
            logger.info("✅ Structure validation passed - all timeframes standardized")
        else:
            logger.warning(f"⚠️ Validation found issues: {issues}")
        
        self.consolidation_report["validation_issues"] = issues
        return final_structure
    
    def generate_report(self) -> str:
        """Generate consolidation report"""
        report_filename = f"timeframe_consolidation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_filename, 'w') as f:
            json.dump(self.consolidation_report, f, indent=2)
        
        logger.info(f"📋 Report saved: {report_filename}")
        return report_filename
    
    def print_summary(self):
        """Print summary of consolidation"""
        print("\n" + "="*60)
        print("📊 TIMEFRAME CONSOLIDATION SUMMARY")
        print("="*60)
        print(f"📁 Files moved: {self.consolidation_report['files_moved']}")
        print(f"🗑️ Folders removed: {self.consolidation_report['folders_removed']}")
        print(f"✅ Actions completed: {len(self.consolidation_report['actions_taken'])}")
        print(f"❌ Errors: {len(self.consolidation_report['errors'])}")
        
        if self.consolidation_report['actions_taken']:
            print(f"\n📋 Actions taken:")
            for action in self.consolidation_report['actions_taken']:
                print(f"   • {action}")
        
        if self.consolidation_report['errors']:
            print(f"\n❌ Errors encountered:")
            for error in self.consolidation_report['errors']:
                print(f"   • {error}")
    
    def run_full_consolidation(self, create_backup: bool = True, dry_run: bool = False) -> bool:
        """
        Run complete timeframe consolidation process
        
        Args:
            create_backup: Create backup before changes
            dry_run: Only analyze, don't make changes
            
        Returns:
            True if successful
        """
        logger.info("🚀 Starting timeframe consolidation...")
        
        # Step 1: Scan current structure
        logger.info("1️⃣ Scanning current structure...")
        structure = self.scan_current_structure()
        
        # Step 2: Identify duplicates
        logger.info("2️⃣ Identifying duplicates...")
        duplicates = self.identify_duplicates(structure)
        
        total_duplicates = sum(len(pairs) for pairs in duplicates.values())
        if total_duplicates == 0:
            logger.info("✅ No duplicates found - structure already standardized!")
            return True
        
        logger.info(f"📋 Found {total_duplicates} folders to consolidate")
        
        if dry_run:
            logger.info("🔍 DRY RUN - No changes will be made")
            for data_type, pairs in duplicates.items():
                for old, new in pairs:
                    logger.info(f"   Would consolidate: {data_type}/{old} -> {data_type}/{new}")
            return True
        
        # Step 3: Consolidate
        logger.info("3️⃣ Consolidating duplicates...")
        success = self.consolidate_timeframes(duplicates, create_backup)
        
        if not success:
            logger.error("❌ Consolidation failed")
            return False
        
        # Step 4: Validate
        logger.info("4️⃣ Validating final structure...")
        self.validate_structure()
        
        # Step 5: Generate report
        logger.info("5️⃣ Generating report...")
        report_file = self.generate_report()
        
        # Step 6: Print summary
        self.print_summary()
        
        logger.info("✅ Timeframe consolidation completed successfully!")
        return True


def main():
    """Main function for script execution"""
    print("\n🔧 TIMEFRAME STANDARDIZATION SCRIPT")
    print("="*50)
    print("Addresses critical folder naming inconsistency:")
    print("• 15min → 15m")
    print("• daily → 1d") 
    print("• 1hour → 1h")
    print("• Prevents data fragmentation")
    print("="*50)
    
    standardizer = TimeframeStandardizer()
    
    # Ask user for confirmation
    print("\n⚠️ This script will modify your data folder structure.")
    response = input("Proceed with analysis? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Operation cancelled")
        return
    
    # First, run dry run to show what would be done
    print("\n🔍 Running analysis (dry run)...")
    standardizer.run_full_consolidation(dry_run=True)
    
    # Ask for confirmation to proceed
    print("\n" + "="*50)
    response = input("Proceed with actual consolidation? (y/N): ").strip().lower()
    
    if response == 'y':
        print("\n🚀 Running consolidation...")
        success = standardizer.run_full_consolidation(create_backup=True, dry_run=False)
        
        if success:
            print("\n🎉 Consolidation completed successfully!")
            print("✅ Timeframe folders now standardized")
            print("✅ Backup created for safety")
            print("🚀 Ready for Phase 3.3 (Risk Management)")
        else:
            print("\n❌ Consolidation failed. Check logs for details.")
    else:
        print("❌ Consolidation cancelled")


if __name__ == "__main__":
    main()