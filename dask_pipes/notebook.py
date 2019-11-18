def get_notebook_path():
    from notebook.notebookapp import list_running_servers
    import ipykernel
    from requests import HTTPError
    import requests
    import re
    import os

    def try_get_notebook_path(kernel_id, url, token):

        r = requests.get(
            url=url + 'api/sessions',
            headers={'Authorization': 'token {}'.format(token), })
        r.raise_for_status()
        response = r.json()
        return {r['kernel']['id']: r['notebook']['path'] for r in response}[kernel_id]

    kernel_id = re.search('kernel-(.*)\.json', ipykernel.connect.get_connection_file()).group(1)
    for rs in list_running_servers():
        try:
            return os.path.join(rs['notebook_dir'], try_get_notebook_path(kernel_id, rs['url'], rs['token']))
        except HTTPError:
            continue
    raise ValueError("Notebook path not available")
