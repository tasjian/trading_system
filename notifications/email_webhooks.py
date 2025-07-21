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
    
    async def test_notification_system(self) -> Dict[str, bool]:
        """Test the notification system with a sample alert."""
        logger.info("Testing notification system...")
        
        # Create test alert
        test_alert = TransactionAlert(
            transaction_id=f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol="AAPL",
            action="buy",
            quantity=10.0,
            price=175.50,
            total_value=1755.00,
            timestamp=datetime.now(),
            portfolio_value=100000.0,
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