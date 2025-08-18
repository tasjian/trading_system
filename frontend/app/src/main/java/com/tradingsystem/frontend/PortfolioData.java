package com.tradingsystem.frontend;

import com.google.gson.annotations.SerializedName;

public class PortfolioData {
    @SerializedName("portfolio_value")
    private double portfolioValue;
    
    @SerializedName("daily_change")
    private double dailyChange;
    
    @SerializedName("daily_change_percent")
    private double dailyChangePercent;
    
    @SerializedName("cash")
    private double cash;
    
    @SerializedName("equity")
    private double equity;
    
    @SerializedName("buying_power")
    private double buyingPower;
    
    @SerializedName("last_updated")
    private String lastUpdated;

    // Constructors
    public PortfolioData() {}

    public PortfolioData(double portfolioValue, double dailyChange, double dailyChangePercent, 
                        double cash, double equity, double buyingPower, String lastUpdated) {
        this.portfolioValue = portfolioValue;
        this.dailyChange = dailyChange;
        this.dailyChangePercent = dailyChangePercent;
        this.cash = cash;
        this.equity = equity;
        this.buyingPower = buyingPower;
        this.lastUpdated = lastUpdated;
    }

    // Getters
    public double getPortfolioValue() { return portfolioValue; }
    public double getDailyChange() { return dailyChange; }
    public double getDailyChangePercent() { return dailyChangePercent; }
    public double getCash() { return cash; }
    public double getEquity() { return equity; }
    public double getBuyingPower() { return buyingPower; }
    public String getLastUpdated() { return lastUpdated; }

    // Setters
    public void setPortfolioValue(double portfolioValue) { this.portfolioValue = portfolioValue; }
    public void setDailyChange(double dailyChange) { this.dailyChange = dailyChange; }
    public void setDailyChangePercent(double dailyChangePercent) { this.dailyChangePercent = dailyChangePercent; }
    public void setCash(double cash) { this.cash = cash; }
    public void setEquity(double equity) { this.equity = equity; }
    public void setBuyingPower(double buyingPower) { this.buyingPower = buyingPower; }
    public void setLastUpdated(String lastUpdated) { this.lastUpdated = lastUpdated; }
}