from torch import Tensor, nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_lstm_layers: int
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_lstm_layers = num_lstm_layers

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_lstm_layers
        )
        self.fc_layer = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, X: Tensor) -> Tensor:
        lstm_output, _ = self.lstm(X)
        model_output = self.fc_layer(lstm_output)

        return model_output

class oLSTM(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int, 
        num_lstm_layers: int,
        dropout: float = 0.2,
        bidirectional: bool = True
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_lstm_layers = num_lstm_layers
        self.num_heads = 4
        
        # Ensure hidden_size is divisible by num_heads when bidirectional
        if bidirectional and hidden_size % self.num_heads != 0:
            self.hidden_size = ((hidden_size + (self.num_heads - 1)) // self.num_heads) * self.num_heads
        
        # Bidirectional LSTM for better temporal pattern recognition
        # - Processes sequence in both directions to capture future and past dependencies
        # - Doubles effective hidden size due to bidirectional concatenation
        # - Applies dropout between layers for regularization when multi-layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism for focusing on relevant parts of the sequence
        # - Uses 4 attention heads to capture different types of patterns
        # - Each head can focus on different aspects (e.g., short vs long-term dependencies)
        # - Helps model identify important temporal relationships in battery behavior
        lstm_output_size = self.hidden_size * 2 if bidirectional else self.hidden_size
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Multiple fully connected layers with skip connections
        # - Allows for deeper feature processing while maintaining gradient flow
        # - Skip connections help preserve important battery state information
        # - Two layers chosen as balance between depth and computational cost
        self.fc_layers = nn.ModuleList([
            nn.Linear(lstm_output_size, lstm_output_size)
            for _ in range(2)
        ])
        
        # Final output transformation and regularization components
        self.output_layer = nn.Linear(lstm_output_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
    def forward(self, X: Tensor) -> Tensor:
        # Ensure input is contiguous in memory
        X = X.contiguous()
        
        # LSTM processing
        lstm_output, _ = self.lstm(X)
        
        # Self-attention mechanism: Focus on relevant temporal patterns
        # Allows model to weigh importance of different timesteps dynamically
        attention_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        
        # Skip connections and layer normalization
        # Combines attention output with original LSTM output to preserve information
        x = self.layer_norm(attention_output + lstm_output)
        
        # Multiple FC layers with residual connections
        # Deeper processing while maintaining gradient flow and feature preservation
        for fc_layer in self.fc_layers:
            residual = x
            x = self.dropout(fc_layer(x))
            x = F.relu(x)  # Non-linear activation fn
            x = self.layer_norm(x + residual)  # Add skip connection and normalize
        
        return self.output_layer(x)


class CellModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_lstm_layers: int,
    ):
        """Initialize Cell Model for battery voltage prediction.
        
        Model combines an LSTM-based overpotential predictor with OCV to estimate total cell voltage.
        Essentially follows basic ECM principle: V_cell = V_ocv + V_overpotential.
        
        Args:
            input_size (int): Number of input features (SOC, current, etc.)
            hidden_size (int): Number of hidden units in LSTM layers
            output_size (int): Output dimension (typically 1 for overpotential prediction)
            num_lstm_layers (int): Number of stacked LSTM layers for temporal learning
        """
        super().__init__()
        self.overpotential_model = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_lstm_layers=num_lstm_layers,
        )

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass of cell model.
        
        Processes input features through LSTM to predict overpotential, + OCV to get final cell voltage.
        
        Args:
            X (Tuple[Tensor, Tensor]): Tuple containing:
                - OCV: Open Circuit Voltage tensor
                - Features: Input features tensor (SOC, current, temperature)
        
        Returns:
            Tensor: Predicted cell voltage combining OCV and predicted overpotential
        """
        OCV, X = X
        overpotential = self.overpotential_model.forward(X) # Predict overpotential using LSTM model initialized in __init__
        overpotential = self.unnormalize_voltage(overpotential) # Unnormalize the overpotential prediction to actual voltage range
        cell_voltage = OCV + overpotential

        return cell_voltage

    def unnormalize_voltage(self, X: Tensor) -> Tensor:
        """Unnormalize model output to actual voltage range.
        
        Converts normalized overpotential back to actual voltage vals.
        -2V -> 0V range chosen based on typical Li-ion cell characteristics where:
        - Negative values represent voltage DROP during DISCHARGE
        - Maximum drop is limited to 2V (arbitrary val, really just something reasonable)
        - Minimum of 0V essentially just means that cell is at equilibrium
        
        Args:
            X (Tensor): Normalized overpotential values from the model
            
        Returns:
            Tensor: Unnormalized overpotential values in actual voltage range
        """
        min_voltage = -2
        max_voltage = 0

        return (X + 1) / 2 * (max_voltage - min_voltage) + min_voltage