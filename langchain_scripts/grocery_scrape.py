import pdb

from bs4 import BeautifulSoup
from requests import request

url: str = "https://flipp.com/weekly_ads"
request(method="get", url=url)
soup = BeautifulSoup(
    markup=request(method="get", url=url).text,
    features="html.parser",
)
pdb.set_trace()
print(soup)
