from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline 



application=Flask(__name__)
app=application


@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/pred',methods=["GET","POST"])
def pred_datapoint():
    if request.method=="GET":
        return render_template("form.html")
    else:
        data=CustomData(
            Delivery_person_Age=float(request.form.get('Delivery_person_Age')), 
            Delivery_person_Ratings=float(request.form.get('Delivery_person_Ratings')),
            Restaurant_latitude=float(request.form.get('Restaurant_latitude')),
            Restaurant_longitude=float(request.form.get('Restaurant_longitude')),
            Delivery_location_latitude=float(request.form.get('Delivery_location_latitude')),
            Delivery_location_longitude=float(request.form.get('Delivery_location_longitude')),
            Order_Date=str(request.form.get('Order_Date')),
            Time_Orderd=str(request.form.get('Time_Orderd')),
            Time_Order_picked=str(request.form.get('Time_Order_picked')),
            Weather_conditions=str(request.form.get('Weather_conditions')),
            Road_traffic_density=str(request.form.get('Road_traffic_density')),
            Vehicle_condition=float(request.form.get('Vehicle_condition')),
            Type_of_order=str(request.form.get('Type_of_order')),
            Type_of_vehicle=str(request.form.get('Type_of_vehicle')),
            multiple_deliveries=float(request.form.get('multiple_deliveries')),  
            Festival=str(request.form.get('Festival')), 
            City=str(request.form.get('City'))
        )

        print("Received form data:")
        for key, value in request.form.items():
            print(f"{key}: {value}")

        print("Data before passing to CustomData:")
        print(data.__dict__)

        final_new_data=data.get_data_as_DataFrame()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        result=pred
        return render_template("results.html",final_result=result)


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True,port=5000)