package com.tradingsystem.frontend;

import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Query;
import java.util.List;

public interface TradingSystemAPI {
    
    @GET("api/portfolio")
    Call<PortfolioData> getPortfolioData();
    
    @GET("api/orders/recent")
    Call<List<Order>> getRecentOrders(@Query("limit") int limit);
    
    @GET("api/health")
    Call<HealthStatus> getHealthStatus();
}