# Modified from Andrei Karpathy's micrograd library
# Référence et source : https://github.com/karpathy/micrograd/

import random


# =====================================================================
# 1. LE MOTEUR AUTOGRAD (engine.py de Karpathy)
# =====================================================================
class Value:
    """Stocke une valeur scalaire unique et son gradient."""

    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), 'puissances int/float uniquement'
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # Tri topologique de tous les enfants dans le graphe
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Application de la règle de la chaîne à rebours
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f'Value(data={self.data:.4f}, grad={self.grad:.4f})'


# =====================================================================
# 2. L'ARCHITECTURE FFN (nn.py de Karpathy)
# =====================================================================
class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class FFN(Module):
    """Feed-Forward Network (Perceptron Multicouche)."""

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        # Toutes les couches cachées utilisent ReLU ; la dernière couche reste linéaire
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# =====================================================================
# 3. EXEMPLE D'ENTRAÎNEMENT DU FFN (Classification binaire -1 / +1)
# =====================================================================
if __name__ == '__main__':
    random.seed(1337)  # La graine favorite d'Andrej Karpathy

    # Jeu de données : 4 exemples à 3 caractéristiques chacun
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # Cibles souhaitées

    # Architecture FFN : 3 entrées -> couche cachée 1 (4 neurones) -> couche cachée 2 (2 neurones) -> 1 sortie
    model = FFN(3, [4, 2, 1])
    print(f'Nombre de paramètres du FFN : {len(model.parameters())}')

    # Boucle d'entraînement (Descente de gradient)
    for step in range(40):
        # 1. Forward pass (Propagation avant)
        ypred = [model(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # 2. Backward pass (Rétropropagation)
        model.zero_grad()
        loss.backward()

        # 3. Update (Mise à jour des poids avec taux d'apprentissage décroissant)
        learning_rate = 0.05 * (1.0 - step / 40)
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if step % 5 == 0 or step == 39:
            print(f'Étape {step:02d} | Loss : {loss.data:.4f}')

    print('\n--- Prédictions finales ---')
    for x, ygt, yp in zip(xs, ys, ypred):
        print(f'Entrée : {x} | Attendu : {ygt:+.1f} | Prédit : {yp.data:+.4f}')
