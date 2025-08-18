package com.tradingsystem.frontend;

import android.graphics.drawable.GradientDrawable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class OrdersAdapter extends RecyclerView.Adapter<OrdersAdapter.OrderViewHolder> {
    
    private List<Order> orders;
    private SimpleDateFormat inputDateFormat;
    private SimpleDateFormat outputTimeFormat;
    
    public OrdersAdapter(List<Order> orders) {
        this.orders = orders;
        // Alpaca datetime format: "2024-08-17T14:30:00.000000Z"
        this.inputDateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'", Locale.US);
        this.outputTimeFormat = new SimpleDateFormat("h:mm a", Locale.US);
    }

    @NonNull
    @Override
    public OrderViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.order_item, parent, false);
        return new OrderViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull OrderViewHolder holder, int position) {
        Order order = orders.get(position);
        
        // Set symbol
        holder.symbolText.setText(order.getSymbol());
        
        // Set side with color
        String side = order.getSide().toUpperCase();
        holder.sideText.setText(side);
        
        // Set side background color
        GradientDrawable sideBackground = (GradientDrawable) holder.sideText.getBackground();
        int sideColor = order.isBuy() ? 
            ContextCompat.getColor(holder.itemView.getContext(), R.color.buy_color) :
            ContextCompat.getColor(holder.itemView.getContext(), R.color.sell_color);
        sideBackground.setColor(sideColor);
        
        // Set order details (quantity and price)
        String details = formatOrderDetails(order);
        holder.detailsText.setText(details);
        
        // Set status with color
        String status = order.getStatus().toUpperCase();
        holder.statusText.setText(status);
        
        // Set status background color
        GradientDrawable statusBackground = (GradientDrawable) holder.statusText.getBackground();
        int statusColor = getStatusColor(holder.itemView.getContext(), status);
        statusBackground.setColor(statusColor);
        
        // Set time
        String timeText = formatTime(order.getCreatedAt());
        holder.timeText.setText(timeText);
    }

    @Override
    public int getItemCount() {
        return orders.size();
    }
    
    private String formatOrderDetails(Order order) {
        StringBuilder details = new StringBuilder();
        
        // Add quantity
        if (order.getQty() != null && !order.getQty().isEmpty()) {
            details.append(order.getQty()).append(" shares");
        }
        
        // Add price information
        if (order.isFilled() && order.getFilledAvgPrice() != null && !order.getFilledAvgPrice().isEmpty()) {
            details.append(" @ $").append(formatPrice(order.getFilledAvgPrice()));
        } else if (order.getLimitPrice() != null && !order.getLimitPrice().isEmpty()) {
            details.append(" @ $").append(formatPrice(order.getLimitPrice())).append(" (limit)");
        }
        
        return details.toString();
    }
    
    private String formatPrice(String priceStr) {
        try {
            double price = Double.parseDouble(priceStr);
            return String.format(Locale.US, "%.2f", price);
        } catch (NumberFormatException e) {
            return priceStr;
        }
    }
    
    private String formatTime(String dateTimeStr) {
        if (dateTimeStr == null || dateTimeStr.isEmpty()) {
            return "";
        }
        
        try {
            Date date = inputDateFormat.parse(dateTimeStr);
            if (date != null) {
                return outputTimeFormat.format(date);
            }
        } catch (ParseException e) {
            // If parsing fails, try to extract just the time portion
            if (dateTimeStr.contains("T")) {
                String timePart = dateTimeStr.split("T")[1];
                if (timePart.contains(".")) {
                    timePart = timePart.split("\\.")[0];
                }
                if (timePart.contains("Z")) {
                    timePart = timePart.replace("Z", "");
                }
                
                try {
                    SimpleDateFormat simpleTimeFormat = new SimpleDateFormat("HH:mm:ss", Locale.US);
                    Date timeDate = simpleTimeFormat.parse(timePart);
                    if (timeDate != null) {
                        return outputTimeFormat.format(timeDate);
                    }
                } catch (ParseException pe) {
                    // Fallback to showing just HH:mm
                    if (timePart.length() >= 5) {
                        return timePart.substring(0, 5);
                    }
                }
            }
        }
        
        return dateTimeStr; // Fallback to original string
    }
    
    private int getStatusColor(android.content.Context context, String status) {
        switch (status.toLowerCase()) {
            case "filled":
                return ContextCompat.getColor(context, R.color.filled_color);
            case "pending":
            case "new":
            case "partially_filled":
                return ContextCompat.getColor(context, R.color.pending_color);
            case "cancelled":
                return ContextCompat.getColor(context, R.color.cancelled_color);
            case "rejected":
                return ContextCompat.getColor(context, R.color.rejected_color);
            default:
                return ContextCompat.getColor(context, R.color.pending_color);
        }
    }
    
    public void updateOrders(List<Order> newOrders) {
        this.orders.clear();
        this.orders.addAll(newOrders);
        notifyDataSetChanged();
    }

    static class OrderViewHolder extends RecyclerView.ViewHolder {
        TextView symbolText;
        TextView sideText;
        TextView detailsText;
        TextView statusText;
        TextView timeText;

        OrderViewHolder(View itemView) {
            super(itemView);
            symbolText = itemView.findViewById(R.id.order_symbol);
            sideText = itemView.findViewById(R.id.order_side);
            detailsText = itemView.findViewById(R.id.order_details);
            statusText = itemView.findViewById(R.id.order_status);
            timeText = itemView.findViewById(R.id.order_time);
        }
    }
}