from sqlalchemy import create_engine, Column, String, Integer, Float
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI
from datetime import datetime
import os

#debugging code block
print("CHECKING ENV...")
print("DATABASE_URL from getenv:", os.getenv("DATABASE_URL"))
print("DATABASE_URL in environ:", os.environ.get("DATABASE_URL"))
print("ALL ENV KEYS:", list(os.environ.keys()))


app = FastAPI()

from Car_Sales_Prediction import generate_next_month_prediction
#DATABASE_URL = os.getenv("postgresql://car_sales_prediction_0hlf_user:VzlatSJSLlGikwRIkz7ECygw5Q1OrEK5@dpg-d6d77da4d50c73ajmbag-a/car_sales_prediction_0hlf")

DATABASE_URL = os.getenv("DATABASE_URL")
print("DATABASE_URL =", DATABASE_URL) #code for debugging
 
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind = engine)
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column (Integer, primary_key = True, index = True)
    predictions_made_on = Column(String)
    predicted_for_month = Column(Integer)
    company = Column(String)
    model = Column(String)
    naive_prediction = Column(Float)
    lr_prediction = Column(Float)
    final_prediction = Column(Float)

Base.metadata.create_all(bind = engine)

def save_predictions_to_db(predictions):
    db = SessionLocal()

    for pred in predictions:
        db_record = Prediction(
            predictions_made_on = pred["predictions_made_on"],
            predicted_for_month = pred["predicted_for_month"],
            company = pred["company"],
            model = pred["model"],
            naive_prediction = pred["naive_prediction"],
            lr_prediction = pred["lr_prediction"],
            final_prediction = pred ["final_prediction"]
        )
        db.add(db_record)
    
    db.commit()
    db.close()
'''
@app.get("/predictions")
def get_predictions():
    
    # -----Debug code------
    return {"message":"API working successfully"}

    # ----- Date Logic to avoid hardcoding-----
    today = datetime.utcnow()
    cutoff_date = today.strftime("%Y-%m-01")

    #-----s prediction engine to predict-----
    predictions = generate_next_month_prediction(cutoff_date)
    save_predictions_to_db(predictions)

    return {
        "meta": {
            "generated_at": today.isoformat(),
            "message": f"{len(predictions)} predictions generated and saved"
        }
    }


'''

@app.get("/predictions")
def get_predictions():
    today = datetime.utcnow()
    cutoff_date = today.strftime("%Y-%m-01")

    predictions = generate_next_month_prediction(cutoff_date)

    return {
        "count": len(predictions)
    }



    #----Summary KPIs (Computed)
    #total_actual = sum(p["actual_sales"] for p in predictions )
    #total_predicted = sum(p["predicted_sales"] for p in predictions)

    #models_within_10 = sum(
    #    1 for p in predictions
    #    if (abs(p["predicted_sales"]- p["actual_sales"])/p["actual_sales"]<= .10)
    #    )
    
    #models_outside_10 = len(predictions) - models_within_10

    #----build summary table dybamically ------

    #summary_table =[]
    #for p in predictions:
    #    error = p["predicted_sales"]-p["actual_sales"]
    #    error_pct = round(error/p["actual_sales"]*100 , 2) 

    #    summary_table.append({
    #        "company": p["company"],
    #        "model": p["model"],
    #        "month": p["month"],
    #        "actual_sales": p["actual_sales"],
    #        "predicted_sales": p["predicted_sales"],
    #        "error": error,
    #        "error_percent": error_pct,
    #        "confidence": p["confidence"],
    #        "review_flag": "Review" if abs(error_pct) > 10 else "ok"
    #        }
    #    )

    #return {
    #    "meta": {
    #        "generated_at": today.isoformat(),
    #        "confidence_rule": "+/-10%"
    #    },
    #    "summary_kpis":{
    #        "total_actual": total_actual,
    #        "total_predicted": total_predicted,
    #        "models_within_10_precent": models_within_10,
    #        "models_outside_10_precent": models_outside_10 
    #    },
    #    "summary_table": summary_table
    #}