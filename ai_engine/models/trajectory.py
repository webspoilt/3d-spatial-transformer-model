import torch
import torch.nn as nn
from typing import Dict, Any

class TrajectoryPredictor(nn.Module):
    """
    Dynamic Obstacle Trajectory Prediction Model.
    Forecasts where humans/forklifts will be 3 seconds into the future
    based on historical 3D spatial sequences.
    """
    def __init__(self, hidden_dim: int = 256, seq_length: int = 10, predict_length: int = 30):
        super().__init__()
        self.seq_length = seq_length
        self.predict_length = predict_length # e.g., 30 frames = 3 seconds at 10hz
        
        # Simple LSTM-based Temporal Encoder
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        # Trajectory decoder outputting 3D coordinates (x, y, z)
        self.decoder = nn.Linear(hidden_dim, 3)
        
    def forward(self, historical_positions: torch.Tensor) -> torch.Tensor:
        """
        historical_positions: Tensor of shape (batch, seq_length, 3)
        Returns forecasted trajectory of shape (batch, predict_length, 3)
        """
        # Encoder
        lstm_out, (h_n, c_n) = self.lstm(historical_positions)
        
        # Take the last hidden state
        last_hidden = h_n[-1] # Shape: (batch, hidden_dim)
        
        # Decode iteratively (autoregressive mock)
        predictions = []
        current_pos = historical_positions[:, -1, :] # Last known pos
        curr_h = last_hidden
        
        # In a real deployed model, we use an autoregressive cell or a full transformer decoder.
        # Here we mock the projection for performance testing.
        for _ in range(self.predict_length):
            # Project hidden state to velocity displacement
            velocity = self.decoder(curr_h) 
            next_pos = current_pos + velocity
            predictions.append(next_pos.unsqueeze(1))
            current_pos = next_pos
            
        return torch.cat(predictions, dim=1) # Shape: (batch, predict_length, 3)
    
    def predict_obstacle(self, obstacle_history: list) -> Dict[str, Any]:
        """Inference wrapper"""
        if not obstacle_history or len(obstacle_history) < self.seq_length:
            return {"status": "insufficient_data"}
            
        # Convert to tensor and run model
        input_tensor = torch.tensor([obstacle_history], dtype=torch.float32)
        with torch.no_grad():
            trajectory = self.forward(input_tensor)
            
        return {
            "status": "success",
            "forecasted_3s_path": trajectory.squeeze().tolist()
        }
