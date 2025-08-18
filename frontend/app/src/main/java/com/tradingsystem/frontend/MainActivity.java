package com.tradingsystem.frontend;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;
import java.util.ArrayList;
import java.util.List;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity implements SwipeRefreshLayout.OnRefreshListener {
    private static final String TAG = "MainActivity";
    
    private TextView portfolioValueText;
    private TextView dailyChangeText;
    private TextView lastUpdatedText;
    private RecyclerView ordersRecyclerView;
    private SwipeRefreshLayout swipeRefreshLayout;
    
    private OrdersAdapter ordersAdapter;
    private List<Order> ordersList;
    private TradingSystemAPI apiService;
    private Handler refreshHandler;
    private Runnable refreshRunnable;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        initViews();
        setupRecyclerView();
        setupAPI();
        setupAutoRefresh();
        
        // Initial data load
        loadData();
    }
    
    private void initViews() {
        portfolioValueText = findViewById(R.id.portfolio_value);
        dailyChangeText = findViewById(R.id.daily_change);
        lastUpdatedText = findViewById(R.id.last_updated);
        ordersRecyclerView = findViewById(R.id.orders_recycler_view);
        swipeRefreshLayout = findViewById(R.id.swipe_refresh_layout);
        
        swipeRefreshLayout.setOnRefreshListener(this);
        swipeRefreshLayout.setColorSchemeResources(
            R.color.colorPrimary,
            R.color.colorPrimaryDark,
            R.color.colorAccent
        );
    }
    
    private void setupRecyclerView() {
        ordersList = new ArrayList<>();
        ordersAdapter = new OrdersAdapter(ordersList);
        ordersRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        ordersRecyclerView.setAdapter(ordersAdapter);
    }
    
    private void setupAPI() {
        apiService = ApiClient.getClient().create(TradingSystemAPI.class);
    }
    
    private void setupAutoRefresh() {
        refreshHandler = new Handler(Looper.getMainLooper());
        refreshRunnable = new Runnable() {
            @Override
            public void run() {
                loadData();
                refreshHandler.postDelayed(this, 30000); // Refresh every 30 seconds
            }
        };
    }
    
    @Override
    protected void onResume() {
        super.onResume();
        startAutoRefresh();
    }
    
    @Override
    protected void onPause() {
        super.onPause();
        stopAutoRefresh();
    }
    
    private void startAutoRefresh() {
        refreshHandler.post(refreshRunnable);
    }
    
    private void stopAutoRefresh() {
        refreshHandler.removeCallbacks(refreshRunnable);
    }
    
    @Override
    public void onRefresh() {
        loadData();
    }
    
    private void loadData() {
        loadPortfolioData();
        loadRecentOrders();
    }
    
    private void loadPortfolioData() {
        Call<PortfolioData> call = apiService.getPortfolioData();
        call.enqueue(new Callback<PortfolioData>() {
            @Override
            public void onResponse(Call<PortfolioData> call, Response<PortfolioData> response) {
                if (response.isSuccessful() && response.body() != null) {
                    updatePortfolioUI(response.body());
                } else {
                    Log.e(TAG, "Portfolio API error: " + response.code());
                    showError("Failed to load portfolio data");
                }
            }
            
            @Override
            public void onFailure(Call<PortfolioData> call, Throwable t) {
                Log.e(TAG, "Portfolio API failure", t);
                showError("Network error loading portfolio");
            }
        });
    }
    
    private void loadRecentOrders() {
        Call<List<Order>> call = apiService.getRecentOrders(10);
        call.enqueue(new Callback<List<Order>>() {
            @Override
            public void onResponse(Call<List<Order>> call, Response<List<Order>> response) {
                swipeRefreshLayout.setRefreshing(false);
                
                if (response.isSuccessful() && response.body() != null) {
                    updateOrdersUI(response.body());
                } else {
                    Log.e(TAG, "Orders API error: " + response.code());
                    showError("Failed to load recent orders");
                }
            }
            
            @Override
            public void onFailure(Call<List<Order>> call, Throwable t) {
                swipeRefreshLayout.setRefreshing(false);
                Log.e(TAG, "Orders API failure", t);
                showError("Network error loading orders");
            }
        });
    }
    
    private void updatePortfolioUI(PortfolioData data) {
        portfolioValueText.setText(String.format("$%.2f", data.getPortfolioValue()));
        
        double dailyChange = data.getDailyChange();
        String changeText = String.format("%+.2f (%.2f%%)", 
            dailyChange, data.getDailyChangePercent());
        
        dailyChangeText.setText(changeText);
        
        // Set color based on change
        int color = dailyChange >= 0 ? 
            getResources().getColor(R.color.profit_green) : 
            getResources().getColor(R.color.loss_red);
        dailyChangeText.setTextColor(color);
        
        lastUpdatedText.setText("Last updated: " + getCurrentTime());
    }
    
    private void updateOrdersUI(List<Order> orders) {
        ordersList.clear();
        ordersList.addAll(orders);
        ordersAdapter.notifyDataSetChanged();
        
        lastUpdatedText.setText("Last updated: " + getCurrentTime());
    }
    
    private void showError(String message) {
        Log.e(TAG, message);
        // Set placeholder text to indicate connection issues
        runOnUiThread(() -> {
            portfolioValueText.setText("Connection Error");
            dailyChangeText.setText(message);
            dailyChangeText.setTextColor(getResources().getColor(R.color.loss_red));
            lastUpdatedText.setText("Check backend connection");
        });
    }
    
    private String getCurrentTime() {
        return new java.text.SimpleDateFormat("HH:mm:ss", 
            java.util.Locale.getDefault()).format(new java.util.Date());
    }
}