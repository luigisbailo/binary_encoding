import torch
import torch.nn as nn



def get_penultimate (pen_lin_nodes, pen_nonlin_nodes, backbone_dense_nodes, num_classes):

    if pen_lin_nodes:
        pen_layer = nn.Linear(in_features=backbone_dense_nodes, out_features=pen_lin_nodes)
        output_layer = nn.Linear(in_features=pen_lin_nodes, out_features=num_classes)
    elif pen_nonlin_nodes:
        pen_layer = nn.Sequential(
            nn.Linear(in_features=backbone_dense_nodes, out_features=pen_nonlin_nodes),
            nn.ReLU()           
        )
        output_layer = nn.Linear(in_features=pen_nonlin_nodes, out_features=num_classes)
    else:
        pen_layer = None
        output_layer = nn.Linear(in_features=backbone_dense_nodes, out_features=num_classes)    

    return pen_layer, output_layer


class MLPvanilla (nn.Module):
   
    def __init__(self, pen_lin_nodes=64, pen_nonlin_nodes=None, dropout=0.5, input_dims=784):
         
        super(MLPvanilla, self).__init__()
        self.pen_lin_nodes = pen_lin_nodes
        self.pen_nonlin_nodes = pen_nonlin_nodes
        self.backbone_dense_nodes = 1024
        self.num_classes = 10
        self.input_dims = input_dims
        self.backbone_dense = nn.Sequential(
            nn.Linear(self.input_dims, self.backbone_dense_nodes),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone_dense_nodes, self.backbone_dense_nodes),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone_dense_nodes, self.backbone_dense_nodes),
            nn.ReLU(),
            )       
        
        self.pen_layer, self.output_layer = get_penultimate(self.pen_lin_nodes, self.pen_nonlin_nodes, self.backbone_dense_nodes, self.num_classes)

        
    def forward(self, x):
        
        x = torch.flatten(x, start_dim=1)
        x = self.backbone_dense(x)
        
        if self.pen_lin_nodes:
            x_pen = self.pen_layer(x)
            x_output = self.output_layer(x_pen)
            y = torch.softmax(x_output, dim=-1)
        elif self.pen_nonlin_nodes:
            x_pen = self.pen_layer(x)
            x_output = self.output_layer(x_pen)
            y = torch.softmax(x_output, dim=-1)
        else:
            x_pen = x
            x_output = self.output_layer(x_pen)
            y = torch.softmax(x_output, dim=-1)
        
        return y.reshape(-1,10), x_pen



class VGG11(nn.Module):
    def __init__(self, pen_lin_nodes = 64, pen_nonlin_nodes = None, dropout=0.5):

        super(VGG11, self).__init__()
        self.in_channels = 1
        self.num_classes = 10
        self.dropout = dropout
        self.backbone_dense_nodes = 1024
        self.pen_lin_nodes = pen_lin_nodes
        self.pen_nonlin_nodes = pen_nonlin_nodes

        self.backbone_conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.backbone_dense_layers = nn.Sequential(
            nn.Linear(in_features=512*3*3, out_features=self.backbone_dense_nodes),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=self.backbone_dense_nodes, out_features=self.backbone_dense_nodes),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=self.backbone_dense_nodes, out_features=self.backbone_dense_nodes),
            nn.ReLU(),
        )
        
        self.pen_layer, self.output_layer = get_penultimate(self.pen_lin_nodes, self.pen_nonlin_nodes, self.backbone_dense_nodes, self.num_classes)
            
        
    def forward(self, x):

        x = self.backbone_conv_layers(x)    
        x = torch.flatten(x, start_dim=1)
        x = self.backbone_dense_layers(x)

        if self.pen_lin_nodes:
            x_pen = self.pen_layer(x)
            x_output = self.output_layer(x_pen)
            y = torch.softmax(x_output, dim=-1)
        elif self.pen_nonlin_nodes:
            x_pen = self.pen_layer(x)
            x_output = self.output_layer(x_pen)
            y = torch.softmax(x_output, dim=-1)
        else:
            x_pen = x
            x_output = self.output_layer(x_pen)
            y = torch.softmax(x_output, dim=-1)
        
        return y.reshape(-1,10), x_pen
    

 

    # class Classifier_cnn (nn.Module):
   
#     def __init__(self, pen_lin_nodes=64, backbone_nodes = [1024]):
#         super(Classifier_cnn, self).__init__()
#         self.pen_lin_nodes = pen_lin_nodes
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)        
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
        
#         self.hidden_layers = nn.ModuleList()      
#         self.hidden_layers.append(nn.Linear(in_features=576, out_features=backbone_nodes[0]))

#         for l in range (1, len(backbone_nodes)):
#             self.hidden_layers.append(nn.Linear(in_features=backbone_nodes[l-1], out_features=backbone_nodes[l]))

#         if pen_lin_nodes:
#             self.pen_layer = nn.Linear(in_features=backbone_nodes[-1], out_features=pen_lin_nodes)
#             self.output_layer = nn.Linear(in_features=pen_lin_nodes, out_features=10)
#         else:
#             self.output_layer = nn.Linear(in_features=backbone_nodes[-1], out_features=10)
        
#         self.activation = nn.SiLU()

#     def forward(self, x):
#         x = self.activation(self.bn1(self.conv1(x)))
#         x = nn.functional.max_pool2d(x, 2)
#         x = self.activation(self.bn2(self.conv2(x)))
#         x = nn.functional.max_pool2d(x, 2)
#         x = self.activation(self.bn3(self.conv3(x)))
#         x = nn.functional.max_pool2d(x, 2)

#         x = x.view(x.size(0), -1)  # Flatten
        
#         for layer in self.hidden_layers:
#             x = self.activation(layer(x))

#         if self.pen_lin_nodes:
#             x_pen = self.pen_layer(x)
#             x_output = self.output_layer(x_pen)
#             y = torch.softmax(x_output, dim=-1)

#         else:
#             x_pen = x
#             x_output = self.output_layer(x_pen)
#             y = torch.softmax(x_output, dim=-1)
        
#         return y.reshape(-1,10), x_pen
    
