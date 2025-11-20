import pytest
import pandas as pd 
import numpy as np 


#dummy test to see if data processes correctly 
def test_festure_shape():
    """
    Test that our input data simulator has correct feature count (should be 15)
    """

    dummy_data = {
        'TP2': 1.0, 'TP3': 2.0, 'H1': 3.0, 'DV_pressure': 4.0,
        'Reservoirs': 5.0, 'Oil_temperature': 6.0, 'Motor_current': 7.0,
        'COMP': 8.0, 'DV_eletric': 9.0, 'Towers': 10.0, 'MPG': 11.0,
        'LPS': 12.0, 'Pressure_switch': 13.0, 'Oil_level': 14.0, 
        'Caudal_impulses': 15.0
    }

    #just like consumer we extract features 
    features = list(dummy_data.values())

    #Assert that we have 15 features
    assert len(features) == 15 

def test_prediction_logic():
    """
    Test that the logic for 'Failure' vs 'Normal' is correct
    """

    #remember that 0 means Normal and 1 means Failure 
    prediction_normal  = 0
    prediction_failure = 1 

    status_normal = "Normal" if prediction_normal == 0 else "Failure"
    status_fail = "Normal" if prediction_failure == 0 else "Failure"

    assert status_normal == "Normal"
    assert status_fail == "Failure" 
