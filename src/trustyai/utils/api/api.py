"""
Server module
"""

# pylint: disable = import-error, too-few-public-methods, assignment-from-no-return
__SUCCESSFUL_IMPORT = True

try:
    from kubernetes import config, dynamic
    from kubernetes.dynamic.exceptions import ResourceNotFoundError
    from kubernetes.client import api_client

except ImportError as e:
    print(
        "Warning: api dependencies not found. "
        "Dependencies can be installed with 'pip install trustyai[api]"
    )
    __SUCCESSFUL_IMPORT = False

if __SUCCESSFUL_IMPORT:

    class TrustyAIApi:
        """
        Gets TrustyAI service information
        """

        def __init__(self):
            try:
                k8s_client = config.load_incluster_config()
            except config.ConfigException:
                k8s_client = config.load_kube_config()
            self.dyn_client = dynamic.DynamicClient(
                api_client.ApiClient(configuration=k8s_client)
            )

        def get_service_route(self, name: str, namespace: str):
            """
            Gets routes for services under a specified namespace
            """
            route_api = self.dyn_client.resources.get(api_version="v1", kind="Route")
            try:
                service = route_api.get(name=name, namespace=namespace)
                return f"https://{service.spec.host}"
            except ResourceNotFoundError:
                return f"Error accessing service {name} in namespace {namespace}."
