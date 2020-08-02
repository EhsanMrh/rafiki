# EhsanMrh
## Get sum of repos's stars from Github

# Import libraries
import requests
from requests.auth import HTTPBasicAuth

auth_info = {
    "username": "ehsanmrh",
    "password": "sem1377"
}

# Set Authentiacation to a account
authentication = HTTPBasicAuth(auth_info["username"], auth_info["password"])

def number_of_guthub_stars(username):
    # Get user data
    data = requests.get('https://api.github.com/users/' + username, auth = authentication)
    data = data.json()

    url = data['repos_url']
    page_no = 1
    repos_data = []
    while (True):
        response = requests.get(url, auth = authentication)
        response = response.json()
        repos_data = repos_data + response
        repos_fetched = len(response)
        print("Total repositories fetched: {}".format(repos_fetched))
        if (repos_fetched == 30):
            page_no = page_no + 1
            url = data['repos_url'] + '?page=' + str(page_no)
        else:
            break

    # Total stars
    stars = 0
    for index, repo in enumerate(repos_data):
        stars += repo['stargazers_count']

    return stars
