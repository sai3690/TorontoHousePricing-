import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy import distance


app = Flask(__name__)
#Load the model 

RandomForestmodel = pickle.load(open('random_forest_model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictbeta')
def homebeta():
    return render_template('homebeta.html')

@app.route('/predict')
def home():
    return render_template('home.html')



@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'})

    data = data['data']
    new_data = np.array(list(data.values())).reshape(1,-1)
    output = RandomForestmodel.predict(new_data)
    return jsonify(output[0])



@app.route('/predictbeta',methods=['POST'])
def predictbeta():
   bathrooms = request.form["bathrooms"]
   parking = request.form["parking"]
   new_bedroom = request.form["new_bedroom"]
   new_sqft = request.form["new_sqft"]
   address = request.form["address"]
   address_series = pd.Series(address)
   geocoder = RateLimiter(Nominatim(user_agent='tutorial').geocode, min_delay_seconds=1)
   location_series = address_series.apply(geocoder)
   Latitude= location_series.apply(lambda loc: loc.latitude if loc else None)
   Longitude = location_series.apply(lambda loc: loc.longitude if loc else None)

   
   
   
   '''Latitude = request.form["Latitude"]
   Longitude = request.form["Longitude"]'''
   type=request.form['type']
   

   if(type=='Att_Row_Twnhouse'):
    Att_Row_Twnhouse = 1
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0 
    Duplex=0
            

   elif(type=='CoOp_Apt'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 1
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0

   elif(type=='Co_Ownership_Apt'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 1
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0


   elif(type=='Comm_Element_Condo'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 1
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0
    
   elif(type=='Condo_Apt'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 1
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0 
    Duplex=0
 
   elif(type=='Condo_Townhouse'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 1
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0

   elif(type=='Det_Condo'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 1
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0

   elif(type=='Detached'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 1
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0

   elif(type=='Fourplex'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 1
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0

   elif(type=='Leasehold_Condo'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 1
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0

   elif(type=='Multiplex'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 1
    Semi_Detached = 0
    Triplex = 0
    Duplex=0

   elif(type=='Semi_Detached'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0
    Semi_Detached = 1
    Triplex = 0
    Duplex=0

   elif(type=='Triplex'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0
    Semi_Detached = 0
    Triplex = 1
    Duplex=0

   elif(type=='Duplex'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0
    Semi_Detached = 0
    Triplex = 0
    Duplex=1

    print([bathrooms, parking,
         new_bedroom, new_sqft,
      Latitude, Longitude, Att_Row_Twnhouse, CoOp_Apt,
       Co_Ownership_Apt, Comm_Element_Condo, Condo_Apt,
       Condo_Townhouse, Det_Condo, Detached, Duplex, Fourplex,
       Leasehold_Condo, Multiplex, Semi_Detached, Triplex])

   output = RandomForestmodel.predict([[bathrooms, parking,
         new_bedroom, new_sqft,
      Latitude, Longitude, Att_Row_Twnhouse, CoOp_Apt,
       Co_Ownership_Apt, Comm_Element_Condo, Condo_Apt,
       Condo_Townhouse, Det_Condo, Detached, Duplex, Fourplex,
       Leasehold_Condo, Multiplex, Semi_Detached, Triplex]])

   return render_template("homebeta.html",prediction_text="The House price prediction is {}".format(int(output)))






@app.route('/predict',methods=['POST'])
def predict():
   bathrooms = request.form["bathrooms"]
   parking = request.form["parking"]
   new_bedroom = request.form["new_bedroom"]
   new_sqft = request.form["new_sqft"]
   
  
  
   '''
   address = request.form["address"]
   address_series = pd.Series(address)
   geocoder = RateLimiter(Nominatim(user_agent='tutorial').geocode, min_delay_seconds=1)
   location_series = address_series.apply(geocoder)
   Latitude= location_series.apply(lambda loc: loc.latitude if loc else None)
   Longitude = location_series.apply(lambda loc: loc.longitude if loc else None)
   '''
   

   
   
   
   Latitude = request.form["Latitude"]
   Longitude = request.form["Longitude"]
   type=request.form['type']
   

   if(type=='Att_Row_Twnhouse'):
    Att_Row_Twnhouse = 1
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0 
    Duplex=0
            

   elif(type=='CoOp_Apt'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 1
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0

   elif(type=='Co_Ownership_Apt'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 1
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0


   elif(type=='Comm_Element_Condo'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 1
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0
    
   elif(type=='Condo_Apt'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 1
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0 
    Duplex=0
 
   elif(type=='Condo_Townhouse'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 1
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0

   elif(type=='Det_Condo'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 1
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0

   elif(type=='Detached'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 1
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0

   elif(type=='Fourplex'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 1
    Leasehold_Condo = 0
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0

   elif(type=='Leasehold_Condo'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 1
    Multiplex = 0 
    Semi_Detached = 0
    Triplex = 0
    Duplex=0

   elif(type=='Multiplex'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 1
    Semi_Detached = 0
    Triplex = 0
    Duplex=0

   elif(type=='Semi_Detached'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0
    Semi_Detached = 1
    Triplex = 0
    Duplex=0

   elif(type=='Triplex'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0
    Semi_Detached = 0
    Triplex = 1
    Duplex=0

   elif(type=='Duplex'):
    Att_Row_Twnhouse = 0
    CoOp_Apt = 0
    Co_Ownership_Apt= 0
    Comm_Element_Condo = 0
    Condo_Apt = 0
    Condo_Townhouse = 0
    Det_Condo = 0
    Detached = 0
    Fourplex = 0
    Leasehold_Condo = 0
    Multiplex = 0
    Semi_Detached = 0
    Triplex = 0
    Duplex=1

    print([bathrooms, parking,
         new_bedroom, new_sqft,
      Latitude, Longitude, Att_Row_Twnhouse, CoOp_Apt,
       Co_Ownership_Apt, Comm_Element_Condo, Condo_Apt,
       Condo_Townhouse, Det_Condo, Detached, Duplex, Fourplex,
       Leasehold_Condo, Multiplex, Semi_Detached, Triplex])

   output = RandomForestmodel.predict([[bathrooms, parking,
         new_bedroom, new_sqft,
      Latitude, Longitude, Att_Row_Twnhouse, CoOp_Apt,
       Co_Ownership_Apt, Comm_Element_Condo, Condo_Apt,
       Condo_Townhouse, Det_Condo, Detached, Duplex, Fourplex,
       Leasehold_Condo, Multiplex, Semi_Detached, Triplex]])

   return render_template("home.html",prediction_text="The House price prediction is {}".format(int(output)))


'''
    output = RandomForestmodel.predict([['bathrooms', 'parking',
         'new_bedroom', 'new_sqft',
      'Latitude', 'Longitude', 'Att/Row/Twnhouse', 'Co-Op Apt',
       'Co-Ownership Apt', 'Comm Element Condo', 'Condo Apt',
       'Condo Townhouse', 'Det Condo', 'Detached', 'Duplex', 'Fourplex',
       'Leasehold Condo', 'Multiplex', 'Semi-Detached', 'Triplex']])

    return render_template("home.html",prediction_text="The House price prediction is {}".format(int(output)))

   response_dict = {
    "bathrooms": float(request.form["bathrooms"]),
    "parking": float(request.form["parking"]),
    "new_bedroom": float(request.form["new_bedroom"]),
    "Latitude": float(request.form["Latitude"]),
    "Longitude": float(request.form["Longitude"]),
    "Att_Row_Twnhouse": float(Att_Row_Twnhouse),
    "CoOp_Apt": float(CoOp_Apt),
    "Co_Ownership_Apt": float(Co_Ownership_Apt),
    "Comm_Element_Condo": float(Comm_Element_Condo),
    "Condo_Apt": float(Condo_Apt),
    "Condo_Townhouse": float(Condo_Townhouse),
    "Det_Condo": float(Det_Condo),
    "Detached": float(Detached),
    "Fourplex": float(Fourplex),
    "Leasehold_Condo": float(Leasehold_Condo),
    "Multiplex": float(Multiplex),
    "Semi_Detached": float(Semi_Detached),
    "Triplex": float(Triplex)
}

    response_array = np.array(list(response_dict.values()))


    data = response_array
    new_data = np.array(list(data.values())).reshape(1,-1)
    output = RandomForestmodel.predict(new_data)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(int(output)))



Working function:
@app.route('/predict',methods=['POST'])
def predict():
    data = request.form.to_dict()
    new_data = np.array(list(data.values())).reshape(1,-1)
    output = RandomForestmodel.predict(new_data)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(int(output)))

'''



if __name__=="__main__":
    app.run(debug=True)

def __getitem__(self, key):
    return self.__dict__[key]
   