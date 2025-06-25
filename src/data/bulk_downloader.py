"""
Bulk Data Download Utility
Download data for multiple symbols and timeframes efficiently
"""
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from tqdm import tqdm
import time

from src.data.fetcher import RateLimitSafeDataFetcher
from src.data.symbol_master import SymbolMaster
from src.api.connection import FyersConnection

logger = logging.getLogger(__name__)

class BulkDataDownloader:
    """Download data for multiple symbols and timeframes efficiently"""
    
    def __init__(self):
        self.connection = FyersConnection()
        self.fetcher = RateLimitSafeDataFetcher(self.connection)
        self.symbol_master = SymbolMaster()
        
        # Default download configuration
        self.default_config = {
            "symbols": ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:RELIANCE-EQ"],
            "timeframes": ["1d", "15m"],
            "daily_years_back": 5,     # 5 years for daily
            "intraday_years_back": 2,  # 2 years for intraday
            "batch_size": 5,           # Process 5 symbols at a time
            "delay_between_batches": 2 # 2 second delay between batches
        }
    
    def create_download_plan(
        self, 
        symbols: List[str] = None, 
        timeframes: List[str] = None,
        custom_config: Dict = None
    ) -> Dict:
        """Create a comprehensive download plan"""
        config = self.default_config.copy()
        if custom_config:
            config.update(custom_config)
        
        symbols = symbols or config["symbols"]
        timeframes = timeframes or config["timeframes"]
        
        plan = {
            "created_at": datetime.now().isoformat(),
            "config": config,
            "download_tasks": []
        }
        
        total_tasks = 0
        total_estimated_time = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Determine years back based on timeframe
                if timeframe == "1d":
                    years_back = config["daily_years_back"]
                    estimated_chunks = years_back // 1  # Rough estimate
                else:
                    years_back = config["intraday_years_back"]  
                    estimated_chunks = (years_back * 365) // 100  # 100-day chunks
                
                # Estimate download time (conservative)
                estimated_time_minutes = estimated_chunks * 0.5  # 30 seconds per chunk
                
                task = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "years_back": years_back,
                    "estimated_chunks": estimated_chunks,
                    "estimated_time_minutes": estimated_time_minutes,
                    "status": "pending"
                }
                
                plan["download_tasks"].append(task)
                total_tasks += 1
                total_estimated_time += estimated_time_minutes
        
        plan["summary"] = {
            "total_tasks": total_tasks,
            "total_symbols": len(symbols),
            "total_timeframes": len(timeframes),
            "estimated_total_time_minutes": round(total_estimated_time, 1),
            "estimated_total_time_hours": round(total_estimated_time / 60, 1)
        }
        
        return plan
    
    def execute_download_plan(self, plan: Dict, save_progress: bool = True) -> Dict:
        """Execute the download plan with progress tracking"""
        if not self.fetcher.connect():
            return {"status": "failed", "error": "Could not connect to API"}
        
        print(f"\n🚀 Starting Bulk Download")
        print(f"📊 {plan['summary']['total_tasks']} tasks for {plan['summary']['total_symbols']} symbols")
        print(f"⏱️  Estimated time: {plan['summary']['estimated_total_time_hours']:.1f} hours")
        print("=" * 60)
        
        results = {
            "started_at": datetime.now().isoformat(),
            "plan": plan,
            "completed_tasks": [],
            "failed_tasks": [],
            "summary": {}
        }
        
        batch_size = plan["config"]["batch_size"]
        tasks = plan["download_tasks"]
        
        # Process in batches
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(tasks) + batch_size - 1) // batch_size
            
            print(f"\n📦 Batch {batch_num}/{total_batches} ({len(batch)} tasks)")
            print("-" * 40)
            
            # Process batch
            for task in tqdm(batch, desc=f"Batch {batch_num}"):
                result = self._download_single_task(task)
                
                if result["status"] == "success":
                    results["completed_tasks"].append(result)
                    print(f"✅ {task['symbol']} {task['timeframe']}: {result['records']} records")
                else:
                    results["failed_tasks"].append(result)
                    print(f"❌ {task['symbol']} {task['timeframe']}: {result['error']}")
            
            # Delay between batches
            if i + batch_size < len(tasks):
                delay = plan["config"]["delay_between_batches"]
                print(f"⏸️  Waiting {delay} seconds before next batch...")
                time.sleep(delay)
        
        # Final summary
        results["completed_at"] = datetime.now().isoformat()
        results["summary"] = {
            "total_attempted": len(tasks),
            "successful": len(results["completed_tasks"]),
            "failed": len(results["failed_tasks"]),
            "success_rate": (len(results["completed_tasks"]) / len(tasks)) * 100,
            "total_records": sum(t.get("records", 0) for t in results["completed_tasks"]),
            "total_time_minutes": (
                datetime.fromisoformat(results["completed_at"]) - 
                datetime.fromisoformat(results["started_at"])
            ).total_seconds() / 60
        }
        
        # Save progress
        if save_progress:
            self._save_download_results(results)
        
        self._print_final_summary(results)
        return results
    
    def _download_single_task(self, task: Dict) -> Dict:
        """Download data for a single task"""
        try:
            start_time = datetime.now()
            
            data = self.fetcher.get_historical_data(
                symbol=task["symbol"],
                timeframe=task["timeframe"],
                years_back=task["years_back"],
                save_to_file=True,
                force_refresh=False  # Use cache if available
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if not data.empty:
                return {
                    "symbol": task["symbol"],
                    "timeframe": task["timeframe"],
                    "status": "success",
                    "records": len(data),
                    "date_range": f"{data.index.min().date()} to {data.index.max().date()}",
                    "duration_seconds": duration,
                    "completed_at": end_time.isoformat()
                }
            else:
                return {
                    "symbol": task["symbol"],
                    "timeframe": task["timeframe"],
                    "status": "failed",
                    "error": "No data retrieved",
                    "duration_seconds": duration
                }
                
        except Exception as e:
            return {
                "symbol": task["symbol"],
                "timeframe": task["timeframe"],
                "status": "failed",
                "error": str(e),
                "duration_seconds": 0
            }
    
    def _save_download_results(self, results: Dict):
        """Save download results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bulk_download_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    def _print_final_summary(self, results: Dict):
        """Print final download summary"""
        summary = results["summary"]
        
        print(f"\n{'='*60}")
        print("📊 BULK DOWNLOAD COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Successful: {summary['successful']}/{summary['total_attempted']} ({summary['success_rate']:.1f}%)")
        print(f"📈 Total Records: {summary['total_records']:,}")
        print(f"⏱️  Total Time: {summary['total_time_minutes']:.1f} minutes")
        
        if results["failed_tasks"]:
            print(f"\n❌ Failed Downloads:")
            for task in results["failed_tasks"][:5]:  # Show first 5 failures
                print(f"   {task['symbol']} {task['timeframe']}: {task['error']}")
    
    def download_popular_symbols(self) -> Dict:
        """Download data for popular symbols with default settings"""
        popular_symbols = [
            "NSE:NIFTY50-INDEX",
            "NSE:NIFTYBANK-INDEX", 
            "NSE:RELIANCE-EQ",
            "NSE:TCS-EQ",
            "NSE:INFY-EQ",
            "NSE:HDFC-EQ",
            "NSE:ICICIBANK-EQ",
            "NSE:SBIN-EQ"
        ]
        
        plan = self.create_download_plan(
            symbols=popular_symbols,
            timeframes=["1d", "15m"],
            custom_config={
                "daily_years_back": 5,
                "intraday_years_back": 2
            }
        )
        
        return self.execute_download_plan(plan)

# Test function
def test_bulk_downloader():
    """Test the bulk downloader"""
    print("🧪 Testing Bulk Data Downloader")
    
    downloader = BulkDataDownloader()
    
    # Create a small test plan
    test_plan = downloader.create_download_plan(
        symbols=["NSE:NIFTY50-INDEX", "NSE:RELIANCE-EQ"],
        timeframes=["1d"],
        custom_config={"daily_years_back": 1, "batch_size": 2}
    )
    
    print("\n📋 Download Plan:")
    print(f"Tasks: {test_plan['summary']['total_tasks']}")
    print(f"Estimated time: {test_plan['summary']['estimated_total_time_minutes']:.1f} minutes")
    
    # Execute the plan
    results = downloader.execute_download_plan(test_plan)
    
    return results

if __name__ == "__main__":
    test_bulk_downloader()