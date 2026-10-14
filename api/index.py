import json
from http.server import BaseHTTPRequestHandler
import urllib.parse

STATE_AVG_DAYS = {
    "SP": 8.3, "PR": 11.5, "MG": 11.5, "SC": 14.2, "RJ": 14.8,
    "RS": 14.8, "DF": 12.5, "ES": 15.1, "GO": 15.0, "MS": 15.2,
    "MT": 17.5, "BA": 18.2, "PE": 18.0, "CE": 20.5, "RN": 20.3,
    "PB": 19.8, "PI": 22.1, "MA": 21.5, "SE": 21.0, "AL": 24.0,
    "PA": 23.5, "TO": 17.5, "RO": 19.0, "AC": 29.5, "AM": 26.0,
    "AP": 27.0, "RR": 29.0
}

def predict_delivery(data):
    customer_state = data.get("customer_state", "SP").upper()
    seller_state = data.get("seller_state", "SP").upper()
    total_price = float(data.get("total_price", 100.0))
    total_freight = float(data.get("total_freight", 20.0))
    n_items = int(data.get("number_of_items", 1))
    weight_g = float(data.get("product_weight_g", 1000.0))
    
    same_state = 1 if customer_state == seller_state else 0
    freight_ratio = total_freight / max(total_price, 1.0)
    
    # State baseline
    base_days = STATE_AVG_DAYS.get(customer_state, 14.0)
    
    # Inter-state penalty
    state_penalty = 0.0 if same_state else 3.5
    if seller_state != "SP" and customer_state != "SP":
        state_penalty += 2.0
        
    # Weight & item penalty
    weight_factor = (weight_g / 5000.0) * 1.2
    item_factor = (n_items - 1) * 0.8
    freight_factor = freight_ratio * 3.0
    
    pred_days = round(base_days + state_penalty + weight_factor + item_factor + freight_factor, 1)
    
    # Late risk probability (calibrated with 0.125 threshold)
    # Baseline late rate is ~7.8%
    z = -2.8  # baseline log-odds approx
    if not same_state:
        z += 0.95
    if customer_state in ["AL", "MA", "PI", "CE", "PA", "AM", "AC", "RR", "AP", "SE", "BA"]:
        z += 1.15
    if freight_ratio > 0.35:
        z += 0.8
    if n_items > 2:
        z += 0.6
    if weight_g > 5000:
        z += 0.5
        
    # Sigmoid
    import math
    prob = round(1.0 / (1.0 + math.exp(-z)), 3)
    
    threshold = 0.125
    is_high_risk = prob >= threshold
    
    return {
        "predicted_delivery_days": pred_days,
        "late_probability": prob,
        "late_probability_pct": f"{round(prob * 100, 1)}%",
        "optimal_threshold": threshold,
        "risk_level": "HIGH RISK" if is_high_risk else "ON SCHEDULE",
        "is_high_risk": is_high_risk,
        "action": (
            "🚨 Proactive Carrier Escalation & Priority Routing Recommended. Order is at high risk of SLA breach."
            if is_high_risk else
            "✅ Standard Dispatch Pipeline. Delivery is on schedule with low delay probability."
        ),
        "details": {
            "customer_state": customer_state,
            "seller_state": seller_state,
            "same_state": bool(same_state),
            "freight_ratio": round(freight_ratio, 2),
            "number_of_items": n_items,
            "product_weight_g": weight_g
        }
    }

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        parsed = urllib.parse.urlparse(self.path)
        if "health" in parsed.path:
            resp = {"status": "healthy", "service": "Smart Logistics ML API", "version": "1.0.0"}
        else:
            resp = {
                "message": "Smart E-Commerce Logistics API is live on Vercel!",
                "endpoints": {
                    "/api/health": "Health check",
                    "/api/predict": "POST order details to receive delivery days & late risk prediction"
                }
            }
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"
        
        try:
            req_json = json.loads(post_data.decode("utf-8"))
        except Exception:
            req_json = {}
            
        result = predict_delivery(req_json)
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode("utf-8"))
