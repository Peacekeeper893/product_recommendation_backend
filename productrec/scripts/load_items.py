
from api.models import product
import csv


def run():
    with open('api/fashion.csv') as file:
        reader = csv.reader(file)
        next(reader)  # Advance past the header


        for row in reader:

            pr = product(product_uid=row[0],product_title=row[7] ,product_img_url = row[8])
            pr.save()
            
