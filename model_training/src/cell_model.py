from torch import Tensor, nn
import torch.nn.functional as F
import torch

class basicLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_lstm_layers: int
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_lstm_layers = num_lstm_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers
        )
        self.fc_layer = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, X: Tensor) -> Tensor:
        lstm_output, _ = self.lstm(X)
        model_output = self.fc_layer(lstm_output)

        return model_output

class TemperatureAwareLSTMLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.temperature_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x: Tensor, temperature: Tensor) -> Tensor:
        # Apply temperature scaling to the output
        lstm_output, _ = self.lstm(x)
        return lstm_output * (1 + self.temperature_scale * temperature)

class LSTM(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int, 
        num_lstm_layers: int,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_temperature_aware: bool = False
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_lstm_layers = num_lstm_layers
        self.use_temperature_aware = use_temperature_aware

        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # Choose LSTM type based on configuration
        if use_temperature_aware:
            self.lstm = TemperatureAwareLSTMLayer(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_lstm_layers
            )
        else:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_lstm_layers,
                # bidirectional=bidirectional
            )
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, X: Tensor) -> Tensor:
        # Normalize input
        X = self.input_norm(X)
        
        if self.use_temperature_aware:
            # Split temperature from other features
            temperature = X[:, :, 2:3]  # Assuming temperature is the third feature
            other_features = X
            
            # Temperature-aware LSTM processing
            lstm_output = self.lstm(other_features, temperature)
        else:
            # Regular LSTM processing
            lstm_output, _ = self.lstm(X)
        
        # Common processing steps
        lstm_output = self.bn(lstm_output.transpose(1, 2)).transpose(1, 2)
        lstm_output = self.dropout(lstm_output)
        
        # Final processing
        x = self.fc1(lstm_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.tanh(x)
        
        return x

class CellModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_lstm_layers: int,
        use_temperature_aware: bool = False
    ):
        """Initialize Cell Model for battery voltage prediction.
        
        Model combines an LSTM-based overpotential predictor with OCV to estimate total cell voltage.
        Essentially follows basic ECM principle: V_cell = V_ocv + V_overpotential.
        
        Args:
            input_size (int): Number of input features (SOC, current, etc.)
            hidden_size (int): Number of hidden units in LSTM layers
            output_size (int): Output dimension (typically 1 for overpotential prediction)
            num_lstm_layers (int): Number of stacked LSTM layers for temporal learning
            use_temperature_aware (bool): Whether to use temperature-aware LSTM layer
        """
        super().__init__()
        self.overpotential_model = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_lstm_layers=num_lstm_layers,
            use_temperature_aware=use_temperature_aware
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