# Trading System Android Frontend

A native Android application that displays real-time portfolio data and recent trading orders from the AI trading system.

## Features

- **Portfolio Overview**: Shows current portfolio value and daily change with color-coded profit/loss indicators
- **Recent Orders**: Displays the 10 most recent orders with detailed information including:
  - Symbol, side (BUY/SELL), quantity, and price
  - Order status with color-coded badges
  - Timestamp of order creation
- **Auto-refresh**: Updates data every 30 seconds automatically
- **Pull-to-refresh**: Manual refresh by pulling down on the screen
- **Material Design**: Modern Android UI with Material Design components

## Building the APK

### Prerequisites

1. **Android Studio** or **Android SDK Command Line Tools**
2. **Java 8 or higher**
3. **Android SDK Platform 34** (API level 34)

### Build Instructions

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/tasjian/trading_system.git
   cd trading_system/frontend
   ```

2. **Set ANDROID_HOME environment variable**:
   ```bash
   export ANDROID_HOME=/path/to/your/android/sdk
   export PATH=$PATH:$ANDROID_HOME/platform-tools:$ANDROID_HOME/tools
   ```

3. **Build the APK**:
   ```bash
   ./gradlew assembleDebug
   ```

4. **Find the generated APK**:
   ```
   app/build/outputs/apk/debug/app-debug.apk
   ```

### Alternative: Build with Android Studio

1. Open Android Studio
2. Choose "Open an existing project"
3. Navigate to the `frontend` directory
4. Let Android Studio sync the project
5. Build → Generate Signed Bundle / APK → APK
6. Choose "debug" build variant
7. Click "Finish"

## Configuration

### Backend API Endpoint

Update the base URL in `ApiClient.java` to point to your trading system backend:

```java
private static final String BASE_URL = "http://your-server:8000/";
```

### Required Backend Endpoints

The app expects the following REST API endpoints:

1. **Portfolio Data**: `GET /api/portfolio`
   ```json
   {
     "portfolio_value": 50000.00,
     "daily_change": 250.50,
     "daily_change_percent": 0.51,
     "cash": 10000.00,
     "equity": 40000.00,
     "buying_power": 20000.00,
     "last_updated": "2024-08-17T14:30:00Z"
   }
   ```

2. **Recent Orders**: `GET /api/orders/recent?limit=10`
   ```json
   [
     {
       "id": "order123",
       "symbol": "AAPL",
       "side": "buy",
       "qty": "10",
       "order_type": "limit",
       "limit_price": "150.00",
       "status": "filled",
       "filled_qty": "10",
       "filled_avg_price": "149.95",
       "created_at": "2024-08-17T14:30:00.000000Z"
     }
   ]
   ```

3. **Health Check**: `GET /api/health`
   ```json
   {
     "status": "healthy",
     "timestamp": "2024-08-17T14:30:00Z",
     "system_health": "operational"
   }
   ```

## Installation

1. Enable "Unknown Sources" in Android Settings → Security
2. Transfer the APK file to your Android device
3. Open the APK file and install
4. Grant any required permissions (Internet access)

## Permissions

The app requires the following permissions:
- `INTERNET`: To communicate with the trading system API
- `ACCESS_NETWORK_STATE`: To check network connectivity

## Technical Details

### Architecture

- **MVVM Pattern**: Clean separation of concerns
- **Retrofit**: HTTP client for API communication
- **RecyclerView**: Efficient list display for orders
- **Material Design Components**: Modern Android UI
- **SwipeRefreshLayout**: Pull-to-refresh functionality

### Key Components

- `MainActivity.java`: Main activity coordinating UI and data
- `TradingSystemAPI.java`: Retrofit interface defining API endpoints
- `OrdersAdapter.java`: RecyclerView adapter for orders list
- `PortfolioData.java`, `Order.java`: Data models matching API responses
- `ApiClient.java`: Retrofit client configuration

### Error Handling

- Network error detection and user feedback
- API error response handling
- Graceful fallback for malformed data
- Connection timeout configuration (30 seconds)

## Troubleshooting

### Common Issues

1. **"Network Error"**: Check that the backend API is running and accessible
2. **"Failed to load data"**: Verify API endpoints are responding correctly
3. **App crashes on startup**: Check Android logs for detailed error messages

### Debug Mode

The app includes HTTP request/response logging in debug builds. Check Android Studio logs for detailed network activity.

### Backend Integration

Ensure your trading system backend provides the required API endpoints. You may need to create a simple REST API wrapper around your existing trading system.

## Security Notes

- This app is designed for development/testing purposes
- Use HTTPS in production environments
- Implement proper authentication for production use
- Consider API rate limiting to prevent abuse

## License

This project is licensed under the MIT License - see the main project LICENSE file for details.