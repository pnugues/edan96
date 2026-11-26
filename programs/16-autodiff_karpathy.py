# Modified from Andrei Karpathy's micrograd library
# Référence et source : https://github.com/karpathy/micrograd/
import random
from graphviz import Digraph


# =====================================================================
# 1. MOTEUR AUTOGRAD (engine.py)
# =====================================================================
class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

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
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f'Value(data={self.data:.4f}, grad={self.grad:.4f})'


# =====================================================================
# 2. ARCHITECTURE FFN (nn.py)
# =====================================================================
class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [
            Value(random.uniform(-1, 1), label=f'w{i}') for i in range(nin)
        ]
        self.b = Value(0.0, label='b')
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

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
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
# 3. VISUALISATION DU GRAPHE (Graphviz)
# =====================================================================
def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format='png', rankdir='LR'):
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        label_str = f'{{ {n.label} | data {n.data:.2f} | grad {n.grad:.2f} }}'
        dot.node(name=uid, label=label_str, shape='record')

        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


# =====================================================================
# 4. EXÉCUTION AVEC MODEL = FFN(3, [4, 4, 1])
# =====================================================================
if __name__ == '__main__':
    random.seed(1337)

    # 3 entrées nommées pour les repérer sur le graphe
    x = [
        Value(2.0, label='x0'),
        Value(3.0, label='x1'),
        Value(-1.0, label='x2'),
    ]
    y_cible = Value(1.0, label='cible')

    # Initialisation demandée : 3 entrées -> 4 -> 2 -> 1 sortie
    model = FFN(3, [4, 2, 1])
    print(f'Nombre de paramètres : {len(model.parameters())}')
    print(f'Paramètres : {model.parameters()}')

    # Passe avant (Forward)
    pred = model(x)
    pred.label = 'pred'
    loss = (pred - y_cible) ** 2
    loss.label = 'loss'

    # Passe arrière (Backward)
    model.zero_grad()
    loss.backward()

    # Génération et ouverture de l'image
    dot = draw_dot(loss)
    dot.render('graphe_ffn_complet', view=True, cleanup=True)
    print("Image 'graphe_ffn_complet.png' générée avec succès !")
