from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base
import datetime

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # Input Features
    koi_period = Column(Float, nullable=True)
    koi_duration = Column(Float, nullable=True)
    koi_depth = Column(Float, nullable=True)
    koi_impact = Column(Float, nullable=True)
    koi_model_snr = Column(Float, nullable=True)
    koi_num_transits = Column(Float, nullable=True)
    koi_ror = Column(Float, nullable=True)
    st_teff = Column(Float, nullable=True)
    st_logg = Column(Float, nullable=True)
    st_met = Column(Float, nullable=True)
    st_mass = Column(Float, nullable=True)
    st_radius = Column(Float, nullable=True)
    st_dens = Column(Float, nullable=True)
    teff_err1 = Column(Float, nullable=True)
    teff_err2 = Column(Float, nullable=True)
    logg_err1 = Column(Float, nullable=True)
    logg_err2 = Column(Float, nullable=True)
    feh_err1 = Column(Float, nullable=True)
    feh_err2 = Column(Float, nullable=True)
    mass_err1 = Column(Float, nullable=True)
    mass_err2 = Column(Float, nullable=True)
    radius_err1 = Column(Float, nullable=True)
    radius_err2 = Column(Float, nullable=True)

    # Outputs
    predicted_radius = Column(Float)
    habitability_class = Column(String)
    confidence = Column(Float)
