"""Email notification system using webhooks for transaction alerts."""

import asyncio
import aiohttp
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl
import os
import pytz
from datetime import datetime, time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class TransactionAlert:
    """Transaction alert data structure."""
    transaction_id: str
    symbol: str
    action: str  # "buy", "sell", "hold"
    quantity: float
    price: float
    total_value: float
    timestamp: datetime
    portfolio_value: float
    confidence: float
    reasoning: str
    agent_source: str
    metadata: Dict[str, Any]

@dataclass
class Position:
    """Position data structure for daily summary."""
    symbol: str
    quantity: float
    market_value: float
    unrealized_pnl: float
    percent_change: float
    side: str  # 'long' or 'short'

@dataclass
class DailySummary:
    """Daily trading summary data structure."""
    date: str
    portfolio_value: float
    cash_balance: float
    total_equity: float
    day_change: float
    day_change_percent: float
    daily_return_percent: float  # Daily rate of return as percentage
    ytd_return_percent: float    # Year-to-date rate of return as percentage
    positions: List[Position]
    total_positions: int
    trades_today: int
    portfolio_performance: str
    risk_metrics: Dict[str, Any]
    market_status: str
    fee_breakdown: Dict[str, Any] = None  # Fee breakdown information

class EmailWebhookNotifier:
    """Email notification system using multiple webhook services."""
    
    def __init__(self):
        self.recipient_email = "ztaschdjian@gmail.com"
        self.webhook_services = {
            "zapier": {
                "enabled": True,
                "url": None,  # Configure with actual Zapier webhook URL
                "method": "POST"
            },
            "make": {
                "enabled": True, 
                "url": None,  # Configure with actual Make.com webhook URL
                "method": "POST"
            },
            "n8n": {
                "enabled": True,
                "url": None,  # Configure with actual n8n webhook URL
                "method": "POST"
            },
            "smtp_gmail": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "port": 587,
                "email": os.getenv("GMAIL_EMAIL", ""),
                "password": os.getenv("GMAIL_APP_PASSWORD", "")  # Use App Password, not regular password
            }
        }
        
        # Load webhook URLs from environment variables
        self.webhook_services["zapier"]["url"] = os.getenv("ZAPIER_WEBHOOK_URL")
        self.webhook_services["make"]["url"] = os.getenv("MAKE_WEBHOOK_URL") 
        self.webhook_services["n8n"]["url"] = os.getenv("N8N_WEBHOOK_URL")
        
    async def send_transaction_notification(self, alert: TransactionAlert) -> Dict[str, bool]:
        """Send transaction notification via all configured methods."""
        logger.info(f"Sending transaction notification for {alert.symbol} {alert.action}")
        
        results = {}
        
        # Prepare email content
        email_subject, email_body = self._generate_email_content(alert)
        
        # Try webhook services first (faster)
        webhook_tasks = []
        for service_name, config in self.webhook_services.items():
            if service_name == "smtp_gmail":
                continue  # Handle SMTP separately
                
            if config["enabled"] and config["url"]:
                webhook_tasks.append(
                    self._send_webhook_notification(service_name, config, alert, email_subject, email_body)
                )
        
        # Execute webhook notifications concurrently
        if webhook_tasks:
            webhook_results = await asyncio.gather(*webhook_tasks, return_exceptions=True)
            for i, result in enumerate(webhook_results):
                service_name = list(self.webhook_services.keys())[i]
                results[service_name] = not isinstance(result, Exception)
                if isinstance(result, Exception):
                    logger.error(f"Webhook {service_name} failed: {result}")
        
        # Try SMTP as fallback
        smtp_success = await self._send_smtp_notification(
            email_subject, email_body, alert
        )
        results["smtp_gmail"] = smtp_success
        
        # Log overall success
        successful_methods = sum(results.values())
        logger.info(f"Transaction notification sent via {successful_methods}/{len(results)} methods")
        
        return results
    
    async def _send_webhook_notification(self, service_name: str, config: Dict, 
                                       alert: TransactionAlert, subject: str, body: str) -> bool:
        """Send notification via webhook service."""
        try:
            payload = {
                "recipient": self.recipient_email,
                "subject": subject,
                "body": body,
                "transaction_data": {
                    "transaction_id": alert.transaction_id,
                    "symbol": alert.symbol,
                    "action": alert.action,
                    "quantity": alert.quantity,
                    "price": alert.price,
                    "total_value": alert.total_value,
                    "timestamp": alert.timestamp.isoformat(),
                    "portfolio_value": alert.portfolio_value,
                    "confidence": alert.confidence,
                    "reasoning": alert.reasoning,
                    "agent_source": alert.agent_source,
                    "metadata": alert.metadata
                },
                "service": "AI Trading System",
                "priority": "high" if alert.total_value > 10000 else "normal"
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    config["method"],
                    config["url"],
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        logger.info(f"✅ {service_name} webhook notification sent successfully")
                        return True
                    else:
                        logger.error(f"❌ {service_name} webhook failed with status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ {service_name} webhook error: {e}")
            return False
    
    async def _send_smtp_notification(self, subject: str, body: str, alert: TransactionAlert) -> bool:
        """Send notification via SMTP (Gmail)."""
        try:
            smtp_config = self.webhook_services["smtp_gmail"]
            
            if not smtp_config["enabled"] or not smtp_config["email"] or not smtp_config["password"]:
                logger.warning("SMTP not configured, skipping email notification")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = smtp_config["email"]
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_config["smtp_server"], smtp_config["port"]) as server:
                server.starttls(context=context)
                server.login(smtp_config["email"], smtp_config["password"])
                server.send_message(msg)
            
            logger.info("✅ SMTP email notification sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ SMTP email notification failed: {e}")
            return False
    
    def _generate_email_content(self, alert: TransactionAlert) -> tuple[str, str]:
        """Generate email subject and HTML body."""
        
        # Determine action emoji and color
        action_info = {
            "buy": {"emoji": "🟢", "color": "#28a745", "verb": "Purchased"},
            "sell": {"emoji": "🔴", "color": "#dc3545", "verb": "Sold"},
            "hold": {"emoji": "🟡", "color": "#ffc107", "verb": "Holding"}
        }
        
        action_data = action_info.get(alert.action.lower(), {"emoji": "🔵", "color": "#007bff", "verb": "Processed"})
        
        # Generate subject
        subject = f"🤖 AI Trading Alert: {action_data['verb']} {alert.symbol} - ${alert.total_value:,.0f}"
        
        # Generate HTML body
        body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>AI Trading System Alert</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f4f4f4; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 30px; }}
                .alert-badge {{ display: inline-block; background: {action_data['color']}; color: white; padding: 10px 20px; border-radius: 20px; font-weight: bold; font-size: 18px; }}
                .transaction-details {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .detail-row {{ display: flex; justify-content: space-between; margin: 10px 0; padding: 8px 0; border-bottom: 1px solid #eee; }}
                .detail-label {{ font-weight: bold; color: #495057; }}
                .detail-value {{ color: #28a745; font-weight: bold; }}
                .reasoning-box {{ background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #2196f3; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; }}
                .confidence-bar {{ background: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden; }}
                .confidence-fill {{ background: linear-gradient(90deg, #28a745 0%, #ffc107 50%, #dc3545 100%); height: 100%; transition: width 0.3s ease; }}
                .portfolio-summary {{ background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%); color: white; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 AI Trading System</h1>
                    <div class="alert-badge">
                        {action_data['emoji']} Transaction Alert
                    </div>
                </div>
                
                <div class="transaction-details">
                    <h2>📊 Transaction Details</h2>
                    
                    <div class="detail-row">
                        <span class="detail-label">Symbol:</span>
                        <span class="detail-value" style="color: {action_data['color']};">{alert.symbol}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Action:</span>
                        <span class="detail-value" style="color: {action_data['color']};">{alert.action.upper()} {action_data['emoji']}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Quantity:</span>
                        <span class="detail-value">{alert.quantity:,.2f} shares</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Price per Share:</span>
                        <span class="detail-value">${alert.price:.2f}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Total Value:</span>
                        <span class="detail-value" style="font-size: 20px;">${alert.total_value:,.2f}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Timestamp:</span>
                        <span class="detail-value">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Agent Source:</span>
                        <span class="detail-value">{alert.agent_source}</span>
                    </div>
                </div>
                
                <div class="portfolio-summary">
                    <h3>💼 Portfolio Impact</h3>
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <strong>Portfolio Value:</strong><br>
                            <span style="font-size: 24px;">${alert.portfolio_value:,.0f}</span>
                        </div>
                        <div>
                            <strong>Confidence Level:</strong><br>
                            <span style="font-size: 24px;">{alert.confidence:.1%}</span>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {alert.confidence:.1%}"></div>
                        </div>
                    </div>
                </div>
                
                <div class="reasoning-box">
                    <h3>🧠 AI Reasoning</h3>
                    <p>{alert.reasoning}</p>
                </div>
                
                <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 20px 0;">
                    <h4>⚠️ Important Notice</h4>
                    <p><strong>This is an automated notification from your AI trading system.</strong></p>
                    <ul>
                        <li>Transaction ID: <code>{alert.transaction_id}</code></li>
                        <li>All trades are executed in paper trading mode for safety</li>
                        <li>Monitor your positions and risk levels regularly</li>
                        <li>Past performance does not guarantee future results</li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p>🤖 Generated by AI Trading System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    <p>This email was sent to: {self.recipient_email}</p>
                    <p style="font-size: 10px;">Trading involves risk. Please trade responsibly.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return subject, body
    
    async def send_portfolio_update_notification(self, portfolio_data: Dict[str, Any]) -> Dict[str, bool]:
        """Send portfolio update notification."""
        logger.info("Sending portfolio update notification")
        
        # Create a portfolio update alert
        alert = TransactionAlert(
            transaction_id=f"PORTFOLIO_UPDATE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol="PORTFOLIO",
            action="update",
            quantity=0.0,
            price=0.0,
            total_value=portfolio_data.get("total_value", 0),
            timestamp=datetime.now(),
            portfolio_value=portfolio_data.get("total_value", 0),
            confidence=portfolio_data.get("confidence", 0.5),
            reasoning=portfolio_data.get("reasoning", "Portfolio rebalancing update"),
            agent_source="portfolio_management_system",
            metadata=portfolio_data
        )
        
        return await self.send_transaction_notification(alert)
    
    def configure_webhooks(self, zapier_url: str = None, make_url: str = None, 
                         n8n_url: str = None, gmail_email: str = None, 
                         gmail_password: str = None) -> None:
        """Configure webhook URLs and SMTP credentials."""
        
        if zapier_url:
            self.webhook_services["zapier"]["url"] = zapier_url
            
        if make_url:
            self.webhook_services["make"]["url"] = make_url
            
        if n8n_url:
            self.webhook_services["n8n"]["url"] = n8n_url
            
        if gmail_email:
            self.webhook_services["smtp_gmail"]["email"] = gmail_email
            
        if gmail_password:
            self.webhook_services["smtp_gmail"]["password"] = gmail_password
        
        logger.info("Webhook configuration updated")
    
    def is_market_closed(self) -> bool:
        """Check if the market is currently closed."""
        est = pytz.timezone('US/Eastern')
        now = datetime.now(est)
        
        # Market is closed on weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return True
        
        # Market hours: 9:30 AM - 4:00 PM EST
        market_open = time(9, 30)  # 9:30 AM
        market_close = time(16, 0)  # 4:00 PM
        
        current_time = now.time()
        
        # Market is closed if current time is outside market hours
        return current_time < market_open or current_time >= market_close
    
    def get_market_status(self) -> str:
        """Get current market status string."""
        if self.is_market_closed():
            est = pytz.timezone('US/Eastern')
            now = datetime.now(est)
            
            if now.weekday() >= 5:  # Weekend
                return "Market Closed (Weekend)"
            else:
                current_time = now.time()
                market_open = time(9, 30)
                
                if current_time < market_open:
                    return "Market Closed (Pre-Market)"
                else:
                    return "Market Closed (After-Hours)"
        else:
            return "Market Open"
    
    async def send_daily_summary(self, summary: DailySummary) -> Dict[str, bool]:
        """Send daily trading summary email at market close."""
        logger.info("Sending daily trading summary")
        
        results = {}
        
        # Generate email content
        subject, body = self._generate_daily_summary_content(summary)
        
        # Try webhook services first (faster)
        webhook_tasks = []
        for service_name, config in self.webhook_services.items():
            if service_name == "smtp_gmail":
                continue  # Handle SMTP separately
                
            if config["enabled"] and config["url"]:
                webhook_tasks.append(
                    self._send_daily_webhook_notification(service_name, config, summary, subject, body)
                )
        
        # Execute webhook notifications concurrently
        if webhook_tasks:
            webhook_results = await asyncio.gather(*webhook_tasks, return_exceptions=True)
            for i, result in enumerate(webhook_results):
                service_name = list(self.webhook_services.keys())[i]
                results[service_name] = not isinstance(result, Exception)
                if isinstance(result, Exception):
                    logger.error(f"Daily summary webhook {service_name} failed: {result}")
        
        # Try SMTP as fallback
        smtp_success = await self._send_daily_smtp_notification(subject, body, summary)
        results["smtp_gmail"] = smtp_success
        
        # Log overall success
        successful_methods = sum(results.values())
        logger.info(f"Daily summary sent via {successful_methods}/{len(results)} methods")
        
        return results
    
    def _generate_daily_summary_content(self, summary: DailySummary) -> tuple[str, str]:
        """Generate daily summary email subject and HTML body."""
        
        # Generate subject
        subject = "algoTrader daily summary"
        
        # Determine performance color and emoji
        if summary.day_change > 0:
            perf_color = "#28a745"  # Green
            perf_emoji = "📈"
            perf_text = "UP"
        elif summary.day_change < 0:
            perf_color = "#dc3545"  # Red  
            perf_emoji = "📉"
            perf_text = "DOWN"
        else:
            perf_color = "#6c757d"  # Gray
            perf_emoji = "➡️"
            perf_text = "FLAT"
        
        # Generate positions table
        positions_html = ""
        if summary.positions:
            for position in summary.positions:
                side_emoji = "🟢" if position.side == "long" else "🔴"
                pnl_color = "#28a745" if position.unrealized_pnl >= 0 else "#dc3545"
                
                positions_html += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;">
                        {side_emoji} {position.symbol}
                    </td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">
                        {position.quantity:,.1f}
                    </td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">
                        ${position.market_value:,.2f}
                    </td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right; color: {pnl_color};">
                        ${position.unrealized_pnl:,.2f}
                    </td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right; color: {pnl_color};">
                        {position.percent_change:+.1f}%
                    </td>
                </tr>
                """
        else:
            positions_html = """
            <tr>
                <td colspan="5" style="padding: 20px; text-align: center; color: #6c757d;">
                    No positions currently held
                </td>
            </tr>
            """
        
        # Generate HTML body
        body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{subject}</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f4;
                }}
                .container {{
                    max-width: 800px;
                    margin: 20px auto;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-align: center;
                    padding: 30px 20px;
                }}
                .content {{
                    padding: 30px;
                }}
                .summary-box {{
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                    border-left: 4px solid #667eea;
                }}
                .performance-box {{
                    background: {perf_color}10;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                    border-left: 4px solid {perf_color};
                }}
                .positions-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    background: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .positions-table th {{
                    background: #f8f9fa;
                    color: #495057;
                    font-weight: 600;
                    padding: 12px 8px;
                    text-align: left;
                    border-bottom: 2px solid #dee2e6;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 15px;
                    text-align: center;
                }}
                .metric-value {{
                    display: block;
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .metric-label {{
                    display: block;
                    font-size: 12px;
                    color: #6c757d;
                    text-transform: uppercase;
                }}
                .footer {{
                    background: #f8f9fa;
                    text-align: center;
                    padding: 20px;
                    color: #6c757d;
                    border-top: 1px solid #dee2e6;
                }}
                .risk-metrics {{
                    background: #fff3cd;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 20px 0;
                    border-left: 4px solid #ffc107;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Daily Trading Summary</h1>
                    <p style="font-size: 18px; margin: 10px 0 0 0;">{summary.date} - {summary.market_status}</p>
                </div>
                
                <div class="content">
                    <div class="performance-box">
                        <h2 style="color: {perf_color}; margin-top: 0;">
                            {perf_emoji} Portfolio Performance: {perf_text}
                        </h2>
                        <div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
                            <div class="metric">
                                <span class="metric-value" style="color: {perf_color};">
                                    ${summary.portfolio_value:,.2f}
                                </span>
                                <span class="metric-label">Portfolio Value</span>
                            </div>
                            <div class="metric">
                                <span class="metric-value" style="color: {perf_color};">
                                    ${summary.day_change:+,.2f}
                                </span>
                                <span class="metric-label">Day Change ($)</span>
                            </div>
                            <div class="metric">
                                <span class="metric-value" style="color: {perf_color};">
                                    {summary.daily_return_percent:+.2f}%
                                </span>
                                <span class="metric-label">Daily Return</span>
                            </div>
                            <div class="metric">
                                <span class="metric-value" style="color: {perf_color if summary.ytd_return_percent >= 0 else '#dc3545'};">
                                    {summary.ytd_return_percent:+.2f}%
                                </span>
                                <span class="metric-label">YTD Return</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="summary-box">
                        <h3>💼 Account Summary</h3>
                        <div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
                            <div class="metric">
                                <span class="metric-value">${summary.total_equity:,.2f}</span>
                                <span class="metric-label">Total Equity</span>
                            </div>
                            <div class="metric">
                                <span class="metric-value">${summary.cash_balance:,.2f}</span>
                                <span class="metric-label">Cash Balance</span>
                            </div>
                            <div class="metric">
                                <span class="metric-value">{summary.total_positions}</span>
                                <span class="metric-label">Active Positions</span>
                            </div>
                            <div class="metric">
                                <span class="metric-value">{summary.trades_today}</span>
                                <span class="metric-label">Trades Today</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="positions-section">
                        <h3>📈 Current Positions</h3>
                        <table class="positions-table">
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th style="text-align: right;">Quantity</th>
                                    <th style="text-align: right;">Market Value</th>
                                    <th style="text-align: right;">Unrealized P&L</th>
                                    <th style="text-align: right;">Change %</th>
                                </tr>
                            </thead>
                            <tbody>
                                {positions_html}
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="risk-metrics">
                        <h3>⚠️ Risk Metrics</h3>
                        <p><strong>Portfolio Performance:</strong> {summary.portfolio_performance}</p>
                        <ul>
                            <li>Maximum position risk: {summary.risk_metrics.get('max_position_risk', 'N/A')}</li>
                            <li>Overall portfolio risk: {summary.risk_metrics.get('portfolio_risk', 'N/A')}</li>
                            <li>Risk assessment: {summary.risk_metrics.get('risk_level', 'Moderate')}</li>
                        </ul>
                    </div>
                    
                    <div class="summary-box">
                        <h3>💰 Trading Cost Analysis</h3>
                        {self._generate_fee_section_html(summary.fee_breakdown)}
                    </div>
                </div>
                
                <div class="footer">
                    <p>🤖 Generated by AI Trading System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    <p>This daily summary was sent to: {self.recipient_email}</p>
                    <p style="font-size: 10px;">Trading involves risk. All trades are executed in paper mode for safety.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return subject, body
    
    def _generate_fee_section_html(self, fee_breakdown: Dict[str, Any]) -> str:
        """Generate HTML section for trading fees."""
        if not fee_breakdown:
            return """
            <div style="text-align: center; color: #28a745; padding: 20px;">
                <h4>📊 Paper Trading - No Fees Charged</h4>
                <p>All trading is conducted in paper mode with zero fees.</p>
            </div>
            """
        
        # Extract fee information
        commission = fee_breakdown.get('commission_fees', 0.0)
        finra_taf = fee_breakdown.get('finra_taf_fees', 0.0)
        finra_cat = fee_breakdown.get('finra_cat_fees', 0.0)
        sec_fees = fee_breakdown.get('sec_fees', 0.0)
        total_fees = fee_breakdown.get('total_fees', 0.0)
        transactions = fee_breakdown.get('transactions_count', 0)
        sell_transactions = fee_breakdown.get('sell_transactions', 0)
        
        # Determine color based on total fees
        fee_color = "#28a745" if total_fees < 1.0 else "#ffc107" if total_fees < 10.0 else "#dc3545"
        
        return f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <div style="display: flex; flex-wrap: wrap; justify-content: space-around; margin-bottom: 15px;">
                <div class="metric">
                    <span class="metric-value" style="color: {fee_color};">
                        ${total_fees:.4f}
                    </span>
                    <span class="metric-label">Total Fees</span>
                </div>
                <div class="metric">
                    <span class="metric-value" style="color: #6c757d;">
                        {transactions}
                    </span>
                    <span class="metric-label">Transactions</span>
                </div>
                <div class="metric">
                    <span class="metric-value" style="color: #6c757d;">
                        {sell_transactions}
                    </span>
                    <span class="metric-label">Sell Orders</span>
                </div>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 6px; margin: 10px 0;">
                <h4 style="margin-top: 0; color: #495057;">🏛️ Alpaca Fee Structure</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px 0; font-weight: 500;">Commission (Stock/ETF):</td>
                        <td style="padding: 8px 0; text-align: right; color: #28a745;">$0.00</td>
                        <td style="padding: 8px 0; text-align: right; color: #6c757d; font-size: 12px;">Commission-free</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px 0; font-weight: 500;">FINRA TAF (Sells):</td>
                        <td style="padding: 8px 0; text-align: right;">${finra_taf:.4f}</td>
                        <td style="padding: 8px 0; text-align: right; color: #6c757d; font-size: 12px;">$0.000166/share</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px 0; font-weight: 500;">FINRA CAT (All):</td>
                        <td style="padding: 8px 0; text-align: right;">${finra_cat:.4f}</td>
                        <td style="padding: 8px 0; text-align: right; color: #6c757d; font-size: 12px;">$0.000046/transaction</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px 0; font-weight: 500;">SEC Fees (Sells):</td>
                        <td style="padding: 8px 0; text-align: right;">${sec_fees:.4f}</td>
                        <td style="padding: 8px 0; text-align: right; color: #6c757d; font-size: 12px;">Regulatory minimum</td>
                    </tr>
                    <tr style="border-top: 2px solid #495057; font-weight: bold;">
                        <td style="padding: 8px 0;">Total Estimated Fees:</td>
                        <td style="padding: 8px 0; text-align: right; color: {fee_color};">${total_fees:.4f}</td>
                        <td style="padding: 8px 0; text-align: right; color: #6c757d; font-size: 12px;">If positions traded</td>
                    </tr>
                </table>
            </div>
            
            <div style="background: #e3f2fd; padding: 12px; border-radius: 6px; border-left: 4px solid #2196f3;">
                <h5 style="margin: 0 0 8px 0; color: #1976d2;">💡 Cost Efficiency Notes</h5>
                <ul style="margin: 0; padding-left: 20px; color: #424242; font-size: 13px;">
                    <li><strong>Paper Trading:</strong> All current trading is fee-free for testing</li>
                    <li><strong>Commission-Free:</strong> Alpaca charges $0 commission on all stock/ETF trades</li>
                    <li><strong>Regulatory Only:</strong> Fees shown are mandatory regulatory pass-through costs</li>
                    <li><strong>Extremely Low Cost:</strong> Total fees typically under $0.10 for entire portfolio</li>
                    <li><strong>Perfect for Algo Trading:</strong> Cost structure ideal for high-frequency strategies</li>
                </ul>
            </div>
        </div>
        """
    
    async def _send_daily_webhook_notification(self, service_name: str, config: Dict, 
                                             summary: DailySummary, subject: str, body: str) -> bool:
        """Send daily summary via webhook service."""
        try:
            payload = {
                "recipient": self.recipient_email,
                "subject": subject,
                "body": body,
                "summary_data": {
                    "date": summary.date,
                    "portfolio_value": summary.portfolio_value,
                    "cash_balance": summary.cash_balance,
                    "total_equity": summary.total_equity,
                    "day_change": summary.day_change,
                    "day_change_percent": summary.day_change_percent,
                    "daily_return_percent": summary.daily_return_percent,
                    "ytd_return_percent": summary.ytd_return_percent,
                    "total_positions": summary.total_positions,
                    "trades_today": summary.trades_today,
                    "portfolio_performance": summary.portfolio_performance,
                    "market_status": summary.market_status,
                    "risk_metrics": summary.risk_metrics
                },
                "service": "AI Trading System - Daily Summary",
                "type": "daily_summary",
                "priority": "normal"
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    config["method"],
                    config["url"],
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        logger.info(f"✅ {service_name} daily summary webhook sent successfully")
                        return True
                    else:
                        logger.error(f"❌ {service_name} daily summary webhook failed with status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ {service_name} daily summary webhook error: {e}")
            return False
    
    async def _send_daily_smtp_notification(self, subject: str, body: str, summary: DailySummary) -> bool:
        """Send daily summary via SMTP."""
        smtp_config = self.webhook_services["smtp_gmail"]
        
        if not smtp_config["enabled"] or not smtp_config["email"] or not smtp_config["password"]:
            logger.warning("SMTP not configured for daily summary, skipping")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = smtp_config["email"]
            msg['To'] = self.recipient_email
            
            # Add HTML body
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            # Create secure SSL context
            context = ssl.create_default_context()
            
            # Send email
            with smtplib.SMTP(smtp_config["smtp_server"], smtp_config["port"]) as server:
                server.starttls(context=context)
                server.login(smtp_config["email"], smtp_config["password"])
                text = msg.as_string()
                server.sendmail(smtp_config["email"], [self.recipient_email], text)
            
            logger.info("✅ Daily summary SMTP email sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Daily summary SMTP email failed: {e}")
            return False
    
    async def test_notification_system(self) -> Dict[str, bool]:
        """Test the notification system with a sample alert."""
        logger.info("Testing notification system...")
        
        # Create test alert
        test_alert = TransactionAlert(
            transaction_id=f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol="SPY",  # Use ETF for testing
            action="buy",
            quantity=10.0,
            price=175.50,
            total_value=1755.00,
            timestamp=datetime.now(),
            portfolio_value=1000.0,
            confidence=0.85,
            reasoning="Test transaction notification from AI Trading System. This is a sample alert to verify the notification system is working correctly.",
            agent_source="test_system",
            metadata={"test": True, "environment": "development"}
        )
        
        results = await self.send_transaction_notification(test_alert)
        
        logger.info(f"Notification test completed. Results: {results}")
        return results

# Global email notifier instance
email_notifier = EmailWebhookNotifier()

async def send_transaction_email(symbol: str, action: str, quantity: float, 
                               price: float, portfolio_value: float, 
                               confidence: float = 0.8, reasoning: str = "",
                               agent_source: str = "ai_trading_system") -> bool:
    """
    Convenience function to send transaction email notification.
    
    Args:
        symbol: Stock symbol
        action: "buy", "sell", or "hold"
        quantity: Number of shares
        price: Price per share
        portfolio_value: Current portfolio value
        confidence: Confidence level (0.0 to 1.0)
        reasoning: Reasoning for the transaction
        agent_source: Source agent/system
        
    Returns:
        bool: True if at least one notification method succeeded
    """
    
    alert = TransactionAlert(
        transaction_id=f"TXN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}",
        symbol=symbol,
        action=action,
        quantity=quantity,
        price=price,
        total_value=quantity * price,
        timestamp=datetime.now(),
        portfolio_value=portfolio_value,
        confidence=confidence,
        reasoning=reasoning,
        agent_source=agent_source,
        metadata={}
    )
    
    results = await email_notifier.send_transaction_notification(alert)
    return any(results.values())

async def send_daily_summary_email(portfolio_value: float, cash_balance: float, 
                                 total_equity: float, day_change: float,
                                 positions_data: List[Dict], trades_today: int = 0,
                                 portfolio_performance: str = "Stable") -> bool:
    """
    Convenience function to send daily summary email.
    
    Args:
        portfolio_value: Current total portfolio value
        cash_balance: Available cash balance
        total_equity: Total equity value
        day_change: Dollar change for the day
        positions_data: List of position dictionaries with symbol, quantity, market_value, etc.
        trades_today: Number of trades executed today
        portfolio_performance: Performance description
        
    Returns:
        bool: True if at least one notification method succeeded
    """
    
    # Import fee calculator
    from utils.fee_calculator import fee_calculator
    
    # Convert positions data to Position objects
    positions = []
    for pos_data in positions_data:
        position = Position(
            symbol=pos_data.get('symbol', ''),
            quantity=pos_data.get('quantity', 0.0),
            market_value=pos_data.get('market_value', 0.0),
            unrealized_pnl=pos_data.get('unrealized_pnl', 0.0),
            percent_change=pos_data.get('percent_change', 0.0),
            side=pos_data.get('side', 'long')
        )
        positions.append(position)
    
    # Calculate fees for positions
    positions_dict = {}
    for pos_data in positions_data:
        symbol = pos_data.get('symbol', '')
        if symbol:
            positions_dict[symbol] = {
                'shares': abs(pos_data.get('quantity', 0)),
                'pnl': pos_data.get('unrealized_pnl', 0.0),
                'side': pos_data.get('side', 'long')
            }
    
    fee_breakdown_obj = fee_calculator.calculate_fees_for_positions(positions_dict)
    
    # Convert fee breakdown to dictionary for template
    fee_breakdown = {
        'commission_fees': fee_breakdown_obj.commission_fees,
        'finra_taf_fees': fee_breakdown_obj.finra_taf_fees,
        'finra_cat_fees': fee_breakdown_obj.finra_cat_fees,
        'sec_fees': fee_breakdown_obj.sec_fees,
        'total_fees': fee_breakdown_obj.total_fees,
        'transactions_count': fee_breakdown_obj.transactions_count,
        'sell_transactions': fee_breakdown_obj.sell_transactions,
        'sell_shares': fee_breakdown_obj.sell_shares
    }
    
    # Calculate day change percentage
    day_change_percent = (day_change / (portfolio_value - day_change)) * 100 if portfolio_value != day_change else 0.0
    
    # Calculate daily return percentage (same as day_change_percent for now)
    daily_return_percent = day_change_percent
    
    # Calculate YTD return percentage (placeholder - would need historical data)
    # For now, use a simple approximation based on current performance
    # In a real implementation, this would compare current portfolio value to start-of-year value
    ytd_return_percent = daily_return_percent * 252  # Approximate using daily return * trading days
    if ytd_return_percent > 100:  # Cap at reasonable maximum
        ytd_return_percent = min(ytd_return_percent, 50.0)
    elif ytd_return_percent < -100:
        ytd_return_percent = max(ytd_return_percent, -50.0)
    
    # Create risk metrics
    risk_metrics = {
        'max_position_risk': f"{max([abs(p.market_value / portfolio_value) * 100 for p in positions] + [0]):.1f}%" if positions else "0.0%",
        'portfolio_risk': f"{abs(day_change / portfolio_value) * 100:.1f}%" if portfolio_value > 0 else "0.0%",
        'risk_level': 'Low' if abs(day_change_percent) < 1 else 'Moderate' if abs(day_change_percent) < 5 else 'High'
    }
    
    # Create daily summary
    summary = DailySummary(
        date=datetime.now().strftime('%Y-%m-%d'),
        portfolio_value=portfolio_value,
        cash_balance=cash_balance,
        total_equity=total_equity,
        day_change=day_change,
        day_change_percent=day_change_percent,
        daily_return_percent=daily_return_percent,
        ytd_return_percent=ytd_return_percent,
        positions=positions,
        total_positions=len(positions),
        trades_today=trades_today,
        portfolio_performance=portfolio_performance,
        risk_metrics=risk_metrics,
        market_status=email_notifier.get_market_status(),
        fee_breakdown=fee_breakdown
    )
    
    results = await email_notifier.send_daily_summary(summary)
    return any(results.values())

if __name__ == "__main__":
    # Test the notification system
    async def main():
        print("🧪 Testing Email Webhook Notification System")
        print("=" * 50)
        
        # Test notification
        results = await email_notifier.test_notification_system()
        
        print(f"📧 Notification Results:")
        for service, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"   {service}: {status}")
        
        successful_services = sum(results.values())
        total_services = len(results)
        
        print(f"\n📊 Overall: {successful_services}/{total_services} services succeeded")
        
        if successful_services > 0:
            print("🎉 Email notification system is working!")
        else:
            print("⚠️ No notification methods succeeded. Check configuration.")
    
    asyncio.run(main())


async def send_batch_transaction_email(batch) -> bool:
    """Send batched transaction email notification."""
    try:
        logger.info(f"Sending batch transaction email for {len(batch.transactions)} transactions")
        
        # Create email notifier
        notifier = EmailWebhookNotifier()
        
        # Generate batch email content
        subject, body = _generate_batch_email_content(batch)
        
        # Send via SMTP (most reliable)
        smtp_config = notifier.webhook_services["smtp_gmail"]
        if smtp_config["enabled"] and smtp_config["email"] and smtp_config["password"]:
            success = await notifier._send_smtp_notification(subject, body, None)
            if success:
                logger.info("✅ Batch transaction email sent successfully")
                return True
        
        logger.warning("❌ Batch transaction email failed - SMTP not configured")
        return False
        
    except Exception as e:
        logger.error(f"Failed to send batch transaction email: {e}")
        return False


def _generate_batch_email_content(batch) -> tuple[str, str]:
    """Generate batch transaction email subject and HTML body."""
    
    summary = batch.get_summary()
    
    # Generate subject
    subject = f"🤖 AI Trading Batch: {summary['total_transactions']} transactions - ${summary['net_value']:,.0f} net"
    
    # Determine net performance color
    net_color = "#28a745" if summary['net_value'] >= 0 else "#dc3545"
    net_emoji = "📈" if summary['net_value'] >= 0 else "📉"
    
    # Generate transactions table
    transactions_html = ""
    for transaction in batch.transactions:
        action_color = "#28a745" if transaction.action.lower() == 'buy' else "#dc3545"
        action_emoji = "🟢" if transaction.action.lower() == 'buy' else "🔴"
        
        transactions_html += f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">
                <strong style="color: {action_color};">{action_emoji} {transaction.symbol}</strong>
            </td>
            <td style="padding: 8px; border-bottom: 1px solid #eee; color: {action_color};">
                {transaction.action.upper()}
            </td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">
                {transaction.quantity:,.0f}
            </td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">
                ${transaction.price:.2f}
            </td>
            <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold;">
                ${transaction.total_value:,.2f}
            </td>
            <td style="padding: 8px; border-bottom: 1px solid #eee; font-size: 12px;">
                {transaction.timestamp.strftime('%H:%M:%S')}
            </td>
        </tr>
        """
    
    # Generate symbols summary
    symbols_list = ', '.join(summary['symbols_traded'])
    
    # Generate HTML body
    body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AI Trading Batch Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f4f4f4; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 30px; }}
            .batch-badge {{ display: inline-block; background: {net_color}; color: white; padding: 12px 24px; border-radius: 25px; font-weight: bold; font-size: 18px; margin: 10px 0; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border: 2px solid #e9ecef; }}
            .summary-card h3 {{ margin: 0 0 10px 0; color: #495057; }}
            .summary-card .value {{ font-size: 24px; font-weight: bold; color: #28a745; }}
            .transactions-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .transactions-table th {{ background: #343a40; color: white; padding: 15px 8px; text-align: left; }}
            .transactions-table td {{ padding: 8px; border-bottom: 1px solid #eee; }}
            .transactions-table tr:hover {{ background: #f8f9fa; }}
            .net-summary {{ background: linear-gradient(45deg, {net_color}22, {net_color}44); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid {net_color}; }}
            .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 AI Trading System</h1>
                <div class="batch-badge">
                    {net_emoji} Transaction Batch Summary
                </div>
                <p style="margin: 10px 0 0 0; opacity: 0.9;">
                    {batch.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {batch.end_time.strftime('%H:%M:%S')} UTC
                </p>
            </div>
            
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>📊 Total Transactions</h3>
                    <div class="value">{summary['total_transactions']}</div>
                </div>
                <div class="summary-card">
                    <h3>🟢 Buy Orders</h3>
                    <div class="value" style="color: #28a745;">{summary['buy_orders']}</div>
                    <div style="font-size: 14px; color: #666;">${summary['total_buy_value']:,.0f}</div>
                </div>
                <div class="summary-card">
                    <h3>🔴 Sell Orders</h3>
                    <div class="value" style="color: #dc3545;">{summary['sell_orders']}</div>
                    <div style="font-size: 14px; color: #666;">${summary['total_sell_value']:,.0f}</div>
                </div>
                <div class="summary-card">
                    <h3>💰 Net Value</h3>
                    <div class="value" style="color: {net_color};">${summary['net_value']:,.0f}</div>
                </div>
            </div>
            
            <div class="net-summary">
                <h3>📈 Batch Summary</h3>
                <p><strong>Symbols Traded:</strong> {symbols_list}</p>
                <p><strong>Time Window:</strong> {summary['duration_seconds']:.0f} seconds</p>
                <p><strong>Portfolio Value:</strong> ${summary['portfolio_value']:,.2f}</p>
                <p><strong>Batch ID:</strong> {batch.batch_id}</p>
            </div>
            
            <h3>📋 Transaction Details</h3>
            <table class="transactions-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Action</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>Total Value</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>
                    {transactions_html}
                </tbody>
            </table>
            
            <div class="footer">
                <p>🤖 Generated by AI Trading System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <p>This batch contained {len(batch.transactions)} transactions executed within {summary['duration_seconds']:.0f} seconds</p>
                <p style="font-size: 10px;">Trading involves risk. All trades are executed in paper mode for safety.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return subject, body