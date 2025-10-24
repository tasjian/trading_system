#!/usr/bin/env python3
"""
Background Data Collection Service

This service runs continuously in the background to pre-fetch market data,
ensuring the RL system never has to wait for API calls during trading hours.

Key Features:
1. Proactive data collection during off-market hours
2. Smart refresh scheduling based on market conditions
3. Health monitoring and automatic recovery
4. Persistent cache with Redis backup
5. Non-blocking operation that never delays trading decisions
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

@dataclass
class DataCollectionJob:
    """Background data collection job."""
    symbols: List[str]
    priority: int  # 1=critical, 5=low
    scheduled_time: datetime
    retry_count: int = 0
    max_retries: int = 3

class BackgroundDataService:
    """Background service for proactive market data collection."""
    
    def __init__(self):
        self.running = False
        self.job_queue = asyncio.PriorityQueue()
        self.completed_jobs = set()
        self.failed_jobs = set()
        self.collection_stats = {
            'jobs_completed': 0,
            'jobs_failed': 0,
            'symbols_collected': 0,
            'cache_hits': 0,
            'last_run': None
        }
        self.worker_tasks = []
        
        # Redis connection for persistent caching
        self.redis_client = None
        self._init_redis()
        
    def _init_redis(self):
        """Initialize Redis connection for persistent caching."""
        try:
            import redis
            self.redis_client = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connected for background data caching")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available for background caching: {e}")
            self.redis_client = None
    
    async def start_service(self):
        """Start the background data collection service."""
        if self.running:
            return
            
        logger.info("🚀 Starting Background Data Collection Service...")
        self.running = True
        
        # Start worker tasks
        self.worker_tasks = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(3)  # 3 concurrent workers
        ]
        
        # Start scheduler task
        scheduler_task = asyncio.create_task(self._scheduler())
        self.worker_tasks.append(scheduler_task)
        
        logger.info("✅ Background Data Service started with 3 workers + scheduler")
    
    async def stop_service(self):
        """Stop the background data collection service."""
        logger.info("🛑 Stopping Background Data Collection Service...")
        self.running = False
        
        # Cancel all tasks
        for task in self.worker_tasks:
            task.cancel()
            
        # Wait for tasks to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            
        self.worker_tasks.clear()
        logger.info("✅ Background Data Service stopped")
    
    async def _scheduler(self):
        """Main scheduler that creates data collection jobs."""
        logger.info("📋 Background data scheduler started")
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Schedule different types of jobs based on time
                if self._is_pre_market_hours():
                    await self._schedule_pre_market_jobs()
                elif self._is_post_market_hours():
                    await self._schedule_post_market_jobs()
                elif self._is_weekend():
                    await self._schedule_weekend_jobs()
                else:
                    # During market hours - only critical updates
                    await self._schedule_critical_jobs()
                
                # Update stats
                self.collection_stats['last_run'] = current_time.isoformat()
                
                # Sleep before next scheduling cycle
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                await asyncio.sleep(60)  # Recovery delay
    
    def _is_pre_market_hours(self) -> bool:
        """Check if it's pre-market hours (4-9:30 AM ET)."""
        now = datetime.now()
        return 4 <= now.hour < 9 or (now.hour == 9 and now.minute < 30)
    
    def _is_post_market_hours(self) -> bool:
        """Check if it's post-market hours (4-8 PM ET)."""
        now = datetime.now()
        return 16 <= now.hour < 20
    
    def _is_weekend(self) -> bool:
        """Check if it's weekend."""
        return datetime.now().weekday() >= 5
    
    async def _schedule_pre_market_jobs(self):
        """Schedule jobs for pre-market hours (data refresh for day ahead)."""
        logger.info("🌅 Scheduling pre-market data collection jobs")
        
        # Get liquid stock symbols for RL system
        symbols = await self._get_liquid_symbols()
        
        # Create high-priority job for most liquid stocks
        priority_symbols = symbols[:50]  # Top 50 most liquid
        job = DataCollectionJob(
            symbols=priority_symbols,
            priority=1,  # Highest priority
            scheduled_time=datetime.now()
        )
        await self.job_queue.put((job.priority, time.time(), job))
        
        # Create medium-priority job for remaining symbols
        remaining_symbols = symbols[50:200]  # Next 150
        if remaining_symbols:
            job = DataCollectionJob(
                symbols=remaining_symbols,
                priority=3,
                scheduled_time=datetime.now() + timedelta(minutes=10)
            )
            await self.job_queue.put((job.priority, time.time(), job))
    
    async def _schedule_post_market_jobs(self):
        """Schedule jobs for post-market hours (update with today's data)."""
        logger.info("🌆 Scheduling post-market data collection jobs")
        
        # Focus on symbols that had high volume today
        active_symbols = await self._get_active_symbols_today()
        
        if active_symbols:
            job = DataCollectionJob(
                symbols=active_symbols,
                priority=2,
                scheduled_time=datetime.now()
            )
            await self.job_queue.put((job.priority, time.time(), job))
    
    async def _schedule_weekend_jobs(self):
        """Schedule comprehensive jobs for weekends."""
        logger.info("🏖️ Scheduling weekend comprehensive data collection")
        
        # Weekend is perfect for comprehensive data collection
        all_symbols = await self._get_all_tradeable_symbols()
        
        # Split into chunks for gradual processing
        chunk_size = 100
        for i in range(0, len(all_symbols), chunk_size):
            chunk_symbols = all_symbols[i:i + chunk_size]
            job = DataCollectionJob(
                symbols=chunk_symbols,
                priority=4,  # Low priority, but comprehensive
                scheduled_time=datetime.now() + timedelta(minutes=i // 10)
            )
            await self.job_queue.put((job.priority, time.time(), job))
    
    async def _schedule_critical_jobs(self):
        """Schedule only critical jobs during market hours."""
        logger.debug("📊 Market hours - only critical updates")
        
        # During market hours, only update if cache is stale
        stale_symbols = await self._get_stale_cached_symbols()
        
        if stale_symbols:
            job = DataCollectionJob(
                symbols=stale_symbols[:20],  # Limit during market hours
                priority=1,
                scheduled_time=datetime.now()
            )
            await self.job_queue.put((job.priority, time.time(), job))
    
    async def _worker(self, worker_name: str):
        """Background worker that processes data collection jobs."""
        logger.info(f"👷 Worker {worker_name} started")
        
        while self.running:
            try:
                # Wait for job with timeout
                try:
                    priority, timestamp, job = await asyncio.wait_for(
                        self.job_queue.get(), timeout=30.0
                    )
                except asyncio.TimeoutError:
                    continue  # No jobs available, continue loop
                
                logger.info(f"🔄 {worker_name} processing {len(job.symbols)} symbols (priority {job.priority})")
                
                # Process the job
                success = await self._execute_data_collection_job(job)
                
                if success:
                    self.completed_jobs.add(id(job))
                    self.collection_stats['jobs_completed'] += 1
                    self.collection_stats['symbols_collected'] += len(job.symbols)
                    logger.info(f"✅ {worker_name} completed job for {len(job.symbols)} symbols")
                else:
                    # Retry failed jobs
                    job.retry_count += 1
                    if job.retry_count < job.max_retries:
                        # Re-queue with lower priority and delay
                        job.priority += 1
                        job.scheduled_time = datetime.now() + timedelta(minutes=5 * job.retry_count)
                        await self.job_queue.put((job.priority, time.time() + 300, job))
                        logger.warning(f"🔄 {worker_name} re-queued failed job (attempt {job.retry_count})")
                    else:
                        self.failed_jobs.add(id(job))
                        self.collection_stats['jobs_failed'] += 1
                        logger.error(f"❌ {worker_name} job failed permanently for {len(job.symbols)} symbols")
                
            except Exception as e:
                logger.error(f"❌ Worker {worker_name} error: {e}")
                await asyncio.sleep(10)  # Recovery delay
        
        logger.info(f"👷 Worker {worker_name} stopped")
    
    async def _execute_data_collection_job(self, job: DataCollectionJob) -> bool:
        """Execute a data collection job."""
        try:
            from tools.optimized_api_client import OptimizedAPIClient
            
            async with OptimizedAPIClient() as api_client:
                # Collect data for all symbols in the job
                market_data = await api_client.get_market_data_batch(job.symbols, days=30)
                
                # Store in Redis cache if available
                if self.redis_client and market_data:
                    await self._cache_market_data(market_data)
                
                # Job successful if we got data for at least 30% of symbols
                success_rate = len(market_data) / len(job.symbols)
                return success_rate >= 0.3
                
        except Exception as e:
            logger.error(f"❌ Job execution failed: {e}")
            return False
    
    async def _cache_market_data(self, market_data: Dict):
        """Cache market data in Redis."""
        try:
            if not self.redis_client:
                return
                
            # Store each symbol's data with expiration
            for symbol, data_points in market_data.items():
                cache_key = f"bg_market_data:{symbol}"
                cache_data = {
                    'data': [
                        {
                            'date': point.date,
                            'open': point.open,
                            'high': point.high,
                            'low': point.low,
                            'close': point.close,
                            'volume': point.volume,
                            'source': point.source
                        } for point in data_points
                    ],
                    'timestamp': time.time()
                }
                
                # Cache for 4 hours
                self.redis_client.setex(
                    cache_key, 
                    4 * 3600, 
                    json.dumps(cache_data)
                )
                
            logger.debug(f"💾 Cached market data for {len(market_data)} symbols in Redis")
            
        except Exception as e:
            logger.error(f"❌ Redis caching error: {e}")
    
    async def get_cached_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Retrieve cached market data for symbols."""
        cached_data = {}
        
        if not self.redis_client:
            return cached_data
            
        try:
            for symbol in symbols:
                cache_key = f"bg_market_data:{symbol}"
                cached_json = self.redis_client.get(cache_key)
                
                if cached_json:
                    cache_data = json.loads(cached_json)
                    # Check if cache is still fresh (within 4 hours)
                    if time.time() - cache_data.get('timestamp', 0) < 4 * 3600:
                        cached_data[symbol] = cache_data['data']
                        self.collection_stats['cache_hits'] += 1
                        
            logger.info(f"📋 Cache provided data for {len(cached_data)}/{len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Cache retrieval error: {e}")
        
        return cached_data
    
    async def _get_liquid_symbols(self) -> List[str]:
        """Get list of liquid symbols for trading."""
        try:
            from tools.alpaca_client import alpaca_client
            
            # Get tradeable assets
            assets = alpaca_client.api.list_assets(status='active', asset_class='us_equity')
            # CRITICAL FIX: Shuffle ALL assets first to eliminate alphabetical bias from Alpaca API
            import random
            assets_list = list(assets)
            random.shuffle(assets_list)
            
            liquid_stocks = []
            
            for asset in assets_list[:1000]:  # Top 1000 by volume after shuffling
                if asset.tradable and asset.shortable:
                    liquid_stocks.append(asset.symbol)
                    
            return liquid_stocks[:500]  # Return top 500 liquid stocks
            
        except Exception as e:
            logger.error(f"❌ Failed to get liquid symbols: {e}")
            # Fallback to common liquid stocks
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ', 'IWM',
                'NFLX', 'AMD', 'INTC', 'JPM', 'BAC', 'WMT', 'DIS', 'PYPL', 'V', 'MA'
            ]
    
    async def _get_active_symbols_today(self) -> List[str]:
        """Get symbols that were particularly active today."""
        # This could be enhanced with volume analysis
        # For now, return top liquid symbols
        return (await self._get_liquid_symbols())[:100]
    
    async def _get_all_tradeable_symbols(self) -> List[str]:
        """Get all tradeable symbols."""
        return await self._get_liquid_symbols()
    
    async def _get_stale_cached_symbols(self) -> List[str]:
        """Get symbols with stale cache that need refresh."""
        stale_symbols = []
        
        if not self.redis_client:
            return stale_symbols
            
        try:
            # Check top liquid symbols for stale cache
            liquid_symbols = await self._get_liquid_symbols()
            
            for symbol in liquid_symbols[:50]:  # Check top 50
                cache_key = f"bg_market_data:{symbol}"
                cached_json = self.redis_client.get(cache_key)
                
                if not cached_json:
                    stale_symbols.append(symbol)
                else:
                    cache_data = json.loads(cached_json)
                    cache_age = time.time() - cache_data.get('timestamp', 0)
                    if cache_age > 2 * 3600:  # Older than 2 hours
                        stale_symbols.append(symbol)
                        
        except Exception as e:
            logger.error(f"❌ Error checking stale cache: {e}")
        
        return stale_symbols
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            'running': self.running,
            'workers_active': len([t for t in self.worker_tasks if not t.done()]),
            'jobs_in_queue': self.job_queue.qsize(),
            'jobs_completed': self.collection_stats['jobs_completed'],
            'jobs_failed': self.collection_stats['jobs_failed'],
            'symbols_collected': self.collection_stats['symbols_collected'],
            'cache_hits': self.collection_stats['cache_hits'],
            'last_run': self.collection_stats['last_run']
        }

# Global background service instance
_background_service = None

def get_background_data_service() -> BackgroundDataService:
    """Get singleton background data service."""
    global _background_service
    if _background_service is None:
        _background_service = BackgroundDataService()
    return _background_service

logger.info("✅ Background Data Service loaded - Proactive data collection ready")