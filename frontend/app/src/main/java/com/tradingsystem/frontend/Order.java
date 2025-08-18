package com.tradingsystem.frontend;

import com.google.gson.annotations.SerializedName;

public class Order {
    @SerializedName("id")
    private String id;
    
    @SerializedName("symbol")
    private String symbol;
    
    @SerializedName("side")
    private String side; // "buy" or "sell"
    
    @SerializedName("qty")
    private String qty;
    
    @SerializedName("order_type")
    private String orderType;
    
    @SerializedName("limit_price")
    private String limitPrice;
    
    @SerializedName("stop_price")
    private String stopPrice;
    
    @SerializedName("status")
    private String status;
    
    @SerializedName("filled_qty")
    private String filledQty;
    
    @SerializedName("filled_avg_price")
    private String filledAvgPrice;
    
    @SerializedName("created_at")
    private String createdAt;
    
    @SerializedName("updated_at")
    private String updatedAt;
    
    @SerializedName("submitted_at")
    private String submittedAt;
    
    @SerializedName("filled_at")
    private String filledAt;

    // Constructors
    public Order() {}

    // Getters
    public String getId() { return id; }
    public String getSymbol() { return symbol; }
    public String getSide() { return side; }
    public String getQty() { return qty; }
    public String getOrderType() { return orderType; }
    public String getLimitPrice() { return limitPrice; }
    public String getStopPrice() { return stopPrice; }
    public String getStatus() { return status; }
    public String getFilledQty() { return filledQty; }
    public String getFilledAvgPrice() { return filledAvgPrice; }
    public String getCreatedAt() { return createdAt; }
    public String getUpdatedAt() { return updatedAt; }
    public String getSubmittedAt() { return submittedAt; }
    public String getFilledAt() { return filledAt; }

    // Setters
    public void setId(String id) { this.id = id; }
    public void setSymbol(String symbol) { this.symbol = symbol; }
    public void setSide(String side) { this.side = side; }
    public void setQty(String qty) { this.qty = qty; }
    public void setOrderType(String orderType) { this.orderType = orderType; }
    public void setLimitPrice(String limitPrice) { this.limitPrice = limitPrice; }
    public void setStopPrice(String stopPrice) { this.stopPrice = stopPrice; }
    public void setStatus(String status) { this.status = status; }
    public void setFilledQty(String filledQty) { this.filledQty = filledQty; }
    public void setFilledAvgPrice(String filledAvgPrice) { this.filledAvgPrice = filledAvgPrice; }
    public void setCreatedAt(String createdAt) { this.createdAt = createdAt; }
    public void setUpdatedAt(String updatedAt) { this.updatedAt = updatedAt; }
    public void setSubmittedAt(String submittedAt) { this.submittedAt = submittedAt; }
    public void setFilledAt(String filledAt) { this.filledAt = filledAt; }
    
    // Utility methods
    public boolean isFilled() {
        return "filled".equalsIgnoreCase(status);
    }
    
    public boolean isBuy() {
        return "buy".equalsIgnoreCase(side);
    }
    
    public boolean isSell() {
        return "sell".equalsIgnoreCase(side);
    }
}