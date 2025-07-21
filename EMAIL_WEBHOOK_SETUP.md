# 📧 Email Webhook Notification Setup Guide

This guide will help you set up email notifications for your AI Trading System transactions sent to `ztaschdjian@gmail.com`.

## 🚀 Quick Setup Options

### Option 1: Gmail SMTP (Recommended - Fastest Setup)

1. **Enable 2-Factor Authentication** on your Gmail account
2. **Generate an App Password**:
   - Go to Google Account Settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
3. **Set Environment Variables**:
   ```bash
   export GMAIL_EMAIL="your-gmail@gmail.com"
   export GMAIL_APP_PASSWORD="your-16-digit-app-password"
   ```

### Option 2: Zapier Webhook (Advanced)

1. **Create Zapier Account** at [zapier.com](https://zapier.com)
2. **Create New Zap**:
   - Trigger: Webhooks by Zapier → Catch Hook
   - Action: Gmail → Send Email
3. **Configure Email Template**:
   - To: `ztaschdjian@gmail.com`
   - Subject: `{{subject}}`
   - Body: `{{body}}`
4. **Get Webhook URL** and set environment variable:
   ```bash
   export ZAPIER_WEBHOOK_URL="https://hooks.zapier.com/hooks/catch/xxxxx/xxxxx/"
   ```

### Option 3: Make.com Webhook

1. **Create Make Account** at [make.com](https://make.com)
2. **Create New Scenario**:
   - Webhook → Custom webhook
   - Gmail → Send an email
3. **Set Environment Variable**:
   ```bash
   export MAKE_WEBHOOK_URL="https://hook.make.com/xxxxx"
   ```

### Option 4: n8n Self-Hosted Webhook

1. **Install n8n**: `npm install n8n -g`
2. **Create Workflow**:
   - Webhook Node → Gmail Node
3. **Set Environment Variable**:
   ```bash
   export N8N_WEBHOOK_URL="http://localhost:5678/webhook/trading-notifications"
   ```

## 🧪 Testing Your Setup

Run the test notification:

```bash
python notifications/email_webhooks.py
```

This will send a test email and show which notification methods succeeded.

## 📨 Email Template Preview

Your transaction notifications will look like this:

**Subject**: `🤖 AI Trading Alert: Purchased AAPL - $1,755`

**Content**:
- 📊 Transaction Details (Symbol, Action, Quantity, Price)
- 💼 Portfolio Impact (Total Value, Confidence Level)
- 🧠 AI Reasoning (Why the trade was made)
- ⚠️ Important Notices (Paper trading, risk warnings)

## 🔧 Integration Points

The email notifications are automatically triggered when:

1. **Any trade is executed** via `alpaca_client.place_order()`
2. **Portfolio rebalancing** occurs
3. **System alerts** are generated

## 📋 Environment Variables Summary

Add these to your `.env` file:

```bash
# Gmail SMTP (Option 1)
GMAIL_EMAIL=your-email@gmail.com
GMAIL_APP_PASSWORD=your-16-digit-password

# Webhook URLs (Options 2-4)
ZAPIER_WEBHOOK_URL=https://hooks.zapier.com/hooks/catch/xxxxx/xxxxx/
MAKE_WEBHOOK_URL=https://hook.make.com/xxxxx
N8N_WEBHOOK_URL=http://localhost:5678/webhook/trading-notifications
```

## 🛠️ Programmatic Configuration

You can also configure webhooks in code:

```python
from notifications.email_webhooks import email_notifier

# Configure webhook URLs
email_notifier.configure_webhooks(
    zapier_url="https://hooks.zapier.com/hooks/catch/xxxxx/xxxxx/",
    gmail_email="your-email@gmail.com",
    gmail_password="your-app-password"
)

# Test the system
results = await email_notifier.test_notification_system()
print(f"Notification test results: {results}")
```

## 🔒 Security Best Practices

1. **Never commit API keys** to version control
2. **Use App Passwords** for Gmail, not your regular password
3. **Set up webhook authentication** where possible
4. **Monitor notification logs** for any issues
5. **Test regularly** to ensure notifications are working

## 📞 Troubleshooting

### Gmail SMTP Issues
- ✅ Ensure 2FA is enabled
- ✅ Use App Password, not regular password
- ✅ Check "Less secure app access" if needed (not recommended)

### Webhook Issues
- ✅ Verify webhook URLs are correct and active
- ✅ Check webhook service logs for errors
- ✅ Test webhooks with sample data first

### No Notifications Received
- ✅ Check spam folder
- ✅ Verify email address `ztaschdjian@gmail.com`
- ✅ Run test notification script
- ✅ Check system logs for errors

## 🚀 You're All Set!

Once configured, you'll receive detailed email notifications for every trade executed by your AI trading system. The notifications include:

- 🎯 **Transaction Details**: What was traded and why
- 📊 **Portfolio Impact**: How it affects your overall portfolio
- 🧠 **AI Reasoning**: The logic behind the trading decision
- ⚠️ **Risk Information**: Important disclaimers and risk notices

Your AI trading system will now keep you informed of every action it takes!