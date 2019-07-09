---
layout: post
title: "Overfitting in Neural Networks"
date: 2019-04-14 12:00 +1000
categories: statistics
excerpt: A brief illustration of the problem of overfitting in neural network classification, showing that dense-er is not always better. The 'Human Activity Recognition' dataset is used, composed of smartphone accelerometer readings from different activities.
---

*In this post, I provide a brief illustration of the problem of overfitting in neural network classification, showing that dense-er is not always better. The 'Human Activity Recognition' dataset is used, composed of smartphone accelerometer readings of individuals performing one of six distinct activities.*

# Overfitting in Neural Network Classification

## Data Exploration

Inspecting the data reveals that 561 continuous-valued predictors are available, as well as an ID variable describing the particular individual performing the activity. This latter variable will be omitted, as we are more interested in generalised activity recognition. Conveniently, all of the remaining predictors have already been scaled to lie between −1 and 1.

```python
> train_data.info()

<class pandas.core.frame.DataFrame>
RangeIndex: 7352 entries, 0 to 7351
Columns: 563 entries, tBodyAcc-mean()-X to Activity
dtypes: float64(561), int64(1), object(1)
memory usage: 31.6+ MB
```

We have around 7,300 training datapoints and 3,000 test datapoints. This represents a relatively small dataset, and should render most computation highly feasible.

```python
> print('training dataset size: {}'.format(train_data.shape))
> print('test dataset size: {}'.format(test_data.shape))

training dataset size: (7352, 563)
test dataset size: (2947, 563)
```

Importantly, the classes are uniformly distributed throughout the data. This will facilitate training, and ensure our model performs equally across all inputs.

![activity_classes](/assets/activity_classes.png){:class="img-responsive" border=0}

## Data Preprocessing

Before we can train a model, some elementary coding of the response variable is required.

```python
le = LabelEncoder()

X_train = train_data.iloc[:,0:(train_data.shape[1]-2)].values
y_train = le.fit_transform(train_data.iloc[:,train_data.shape[1]-1].values)

X_test = test_data.iloc[:,0:(test_data.shape[1]-2)].values
y_test = le.transform(test_data.iloc[:,test_data.shape[1]-1].values)
```

PCA reveals that the first 100 principal components explain 97% of the variance in the data. However, since the dataset is already relatively small, we will preserve all 561 features for modelling purposes.

![activity_PCA](/assets/activity_PCA.png){:class="img-responsive" border=0}

Before proceeding, let's experiment with a t-SNE embedding.

```python
tsne = TSNE()
X_reduced = tsne.fit_transform(X_train)
```

![activity_tsne](/assets/activity_tsne.png){:class="img-responsive" border=0}

This t-SNE transformation suggests that some points should be simple to classify, while others are almost indistinguishable from similar activities. We should not be surprised: sitting and standing are likely to generate almost identical accelerometer readings. Nevertheless, let's see how modelling fares.

## Neural Network Models

We now build a neural network clasification model, using varying architectures.

```python
train_dataset = TensorDataset(torch.from_numpy(X_train), 
                              	torch.from_numpy(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(torch.from_numpy(X_test), 
                             		torch.from_numpy(y_test))
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
```

It will help to write a simple `Torch` class that takes a list of layer sizes as an argument, and constructs a corresponding network with batch normalisation and dropout for each layer. We can also write a function to randomly generate architectures given a number of layers (or to generate an entirely random network).

```python
class QuickModel(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        
        self.layer_sizes = layer_sizes
        
        structure = []
        for i in range(len(layer_sizes)):
            if i == len(layer_sizes) - 2:
                structure.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                break
            else:
                structure.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                structure.append(nn.BatchNorm1d(layer_sizes[i+1]))
                structure.append(nn.ReLU())
                structure.append(nn.Dropout(0.5))
                
        self.layers = nn.ModuleList(structure)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def generateRandomArchitecture(num_layers=None, max_layers=10):
    np.random.seed(42)
    if num_layers is None:
        if max_layers <= 3:
            max_layers = 4
        num_layers = np.random.randint(low=3, high=max_layers)
    layer_sizes = []
    for i in range(num_layers):
        if i == 0:
            layer_sizes.append(561)
        elif i == num_layers - 1:
            layer_sizes.append(6)
        else:
            layer_max = min(int(layer_sizes[-1] * 1.5),2000)
            layer_min = int(layer_sizes[-1] / 3)
            if layer_min < 6:
                layer_size = 6
            else:
                layer_size = np.random.randint(low=layer_min, high=layer_max)
            layer_sizes.append(layer_size)
    return layer_sizes, num_layers
```

Now we can instantiate a set of models to experiment with.

```python
> nets = []

> nets.append(QuickModel([561, 250, 6]).double())
> nets.append(QuickModel([561, 124, 32, 16, 6]).double())
> nets.append(QuickModel([561, 256, 256, 124, 64, 32, 16, 6]).double())
> nets.append(QuickModel(generateRandomArchitecture(num_layers=12)[0]).double())
> nets.append(QuickModel(generateRandomArchitecture(num_layers=15)[0]).double())
> nets.append(QuickModel(generateRandomArchitecture(num_layers=20)[0]).double())

> n_nets = len(nets)
> nets_details = [(len(net.layer_sizes), net.layer_sizes) for net in nets]

> for ii, net in enumerate(nets_details):
>     print("Net {}: {} layers - {}".format(ii+1, net[0], net[1]))

Net 1: 3 layers - [561, 250, 6]
Net 2: 5 layers - [561, 124, 32, 16, 6]
Net 3: 8 layers - [561, 256, 256, 124, 64, 32, 16, 6]
Net 4: 12 layers - [561, 289, 366, 228, 147, 69, 
                    97, 106, 122, 156, 151, 6]
Net 5: 15 layers - [561, 289, 366, 228, 147, 69, 
                    97, 106, 122, 156, 151, 153, 202, 197, 6]
Net 6: 20 layers - [561, 289, 366, 228, 147, 69, 
                    97, 106, 122, 156, 151, 153, 202, 197, 
                    214, 123, 42, 37, 41, 6]
```

Having done so, we then run the training loop, updating the entire set of models concurrently.

```python
for net in nets:
    net.to(device)

criterion = nn.CrossEntropyLoss()
optimizers = [optim.SGD(net.parameters(), lr=0.003, momentum=0.9) 
              	for net in nets]

epochs = 800
train_losses = [[] for net in nets]
test_losses = [[] for net in nets]
test_accuracies = [[] for net in nets]
ensemble_accuracies, ensemble_aucs = [], []

best = [{} for net in nets]

for e in tqdm(range(epochs)):
    for net in nets:
        net.train()
    
    for ii, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        for optimizer in optimizers:
            optimizer.zero_grad()

        logits = [net(inputs) for net in nets]
        losses = [criterion(logit, labels) for logit in logits]
        
        for loss in losses:
            loss.backward()
        
        for optimizer in optimizers:
            optimizer.step()
        
        for i in range(n_nets):
            train_losses[i].append(losses[i].item() / inputs.shape[0])
    
    test_loss = [0 for net in nets]
    accuracy = [0 for net in nets]
    
    for net in nets:
        net.eval()    
    for ii, (inputs, labels) in enumerate(test_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        with torch.no_grad():
            logits = [net(inputs) for net in nets]
            losses = [criterion(logit, labels) for logit in logits]
            
            for i in range(n_nets):
                test_loss[i] += losses[i].item() / inputs.shape[0]
            
            preds = [logit.argmax(dim=1) for logit in logits]
            for i in range(n_nets):
                correct = (preds[i] == labels).type(torch.FloatTensor)
                accuracy[i] += torch.mean(correct).item()
    
    for i in range(n_nets):
        if e == 0:
            best[i]['epoch'] = e
            best[i]['state_dict'] = nets[i].state_dict()
            best[i]['accuracy'] = accuracy[i] / len(test_dataloader)
        elif accuracy[i] / len(test_dataloader) > max(test_accuracies[i]):
            best[i]['epoch'] = e
            best[i]['state_dict'] = nets[i].state_dict()
            best[i]['accuracy'] = accuracy[i] / len(test_dataloader)
        
        test_losses[i].append(test_loss[i] / len(test_dataloader))
        test_accuracies[i].append(accuracy[i] / len(test_dataloader))
```

The results of the training algorithm are visualised below, after first smoothing the training losses into an average over 250-batch sequences.

```python
train_losses_averaged = [[] for net in nets]
interval = 250
for n in range(len(train_losses)):
    for i in range(0,len(train_losses[n]),interval):
        try:
            train_losses_averaged[n].append(
              np.mean(train_losses[n][i:i+interval])
            )
        except IndexError:
            train_losses_averaged[n].append(
              np.mean(train_losses[n][i:])
            )
```

![activity_training_results](/assets/activity_training_results.png){:class="img-responsive" border=0}

Looking at the training loss curves, we see starkly different profiles for each of the models. The simplest model trained the fastest, as would be expected, while the most complex model had relatively little success in fitting the training data. It is also interesting to note that several models plateauded for hundreds of epochs before jumping up in accuracy again. Clearly one must be careful about terminating the training algorithm too early! Overall however, it was the simplest models that appear to have performed best. This is confirmed by checking the maximum test accuracy attained by each model.

```python
> for i in range(n_nets):
>     print("Net {}: {} layers, maximum test accuracy of {:.4f}".format(
>         i+1, nets_details[i][0], best[i]['accuracy']))

Net 1: 3 layers, maximum test accuracy of 0.9758
Net 2: 5 layers, maximum test accuracy of 0.9724
Net 3: 8 layers, maximum test accuracy of 0.8518
Net 4: 12 layers, maximum test accuracy of 0.9721
Net 5: 15 layers, maximum test accuracy of 0.8266
Net 6: 20 layers, maximum test accuracy of 0.3569
```

This is clear evidence that dense models are a poor choice for the activities classification problem. But what has gone wrong with these complex models, and why haven't our simple models been able to reach 100% accuracy?

```python
for i in range(n_nets):
    nets[i].load_state_dict(best[i]['state_dict'])
```

## What's Going Wrong?

Let's investigate further the points that our models are misclassifying with another t-SNE embedding.

![activity_misclassification](/assets/activity_misclassification.png){:class="img-responsive" border=0}

It's clear that all models struggle to differentiate between sitting and standing, and this may well be a fundamental limitation of the dataset. Note however that the simple models have nevertheless correctly identified many instances of sitting and standing, even where the two classes appear to overlay one another. It is likely that in the original 561-dimensional space, a simple hyperplane suffices for distinguishing the majority of sitting and standing observations, and that, thanks to its overly-complex decision boundary, the densest network was unable to settle upon this simple approximation.

Otherwise, the models attain similar performance on the remaining activities. Laying should be easy for any competent network to recognise, and the general activity of 'walking' (upstairs, downstairs, or regular) is classified about as well. Further disparities arise in the delineation of different types of walking; once again, the simpler networks, with necessarily cruder approximations of the optimal decision boundary, perform better. It is interesting to note that the denser networks appear to have almost perfectly classified regular walking, yet have entirely failed to distinguish walking up and down stairs. My guess would be that this is similarly due to a messy, convoluted decision boundary that never achieves a perfect split, and so consistently remains terrible.

A brief aside regarding ensemble models: it may seem in this case that an ensemble of the three highest-performing models could offer a boost in accuracy. This is however not the case, and looking at the above t-SNE embeddings, it is easy to see why. All three of the models misclassify the same type of datapoint, meaning that no increase in performance is possible by combining their outputs in a voting arrangement. If all of your ensemble members are always wrong in the same way, the average of their predictions will be just as wrong - one would be better off investing effort in training a single good classifier. This contrasts with the randomness that may arise in other contexts, where different members of an ensemble may be biased in different directions.

The moral of the story: denser is not always better.