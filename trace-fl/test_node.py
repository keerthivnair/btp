import flwr as fl
from flwr.clientapp import ClientApp
from flwr.common import Context

def client_fn(context: Context):
    print("NODE CONFIG:", context.node_config)
    print("NODE ID:", context.node_id)
    return fl.client.NumPyClient().to_client()

app = ClientApp(client_fn=client_fn)
