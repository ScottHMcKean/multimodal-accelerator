def index_exists(client, vs_endpoint, index_name):
    try:
        index_info = client.get_index(vs_endpoint, index_name)
        return True
    except Exception as e:
        if 'IndexNotFoundException' in str(e):
            return False
        else:
            raise e