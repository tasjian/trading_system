package com.tradingsystem.frontend;

import com.google.gson.annotations.SerializedName;

public class HealthStatus {
    @SerializedName("status")
    private String status;
    
    @SerializedName("timestamp")
    private String timestamp;
    
    @SerializedName("system_health")
    private String systemHealth;

    // Constructors
    public HealthStatus() {}

    public HealthStatus(String status, String timestamp, String systemHealth) {
        this.status = status;
        this.timestamp = timestamp;
        this.systemHealth = systemHealth;
    }

    // Getters
    public String getStatus() { return status; }
    public String getTimestamp() { return timestamp; }
    public String getSystemHealth() { return systemHealth; }

    // Setters
    public void setStatus(String status) { this.status = status; }
    public void setTimestamp(String timestamp) { this.timestamp = timestamp; }
    public void setSystemHealth(String systemHealth) { this.systemHealth = systemHealth; }
}