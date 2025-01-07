import os
import logging
from contextlib import asynccontextmanager
from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up application...")
    logger.info("Checking environment variables...")
    if not all([os.getenv('OPENAI_API_KEY'), os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY')]):
        logger.error("Missing required environment variables!")
    else:
        logger.info("All required environment variables are set")
    yield
    # Shutdown
    logger.info("Shutting down application...")

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

class DetailedOpportunities(BaseModel):
    detailed_opportunities: str

class OpportunitiesResponse(BaseModel):
    brand_name: str
    detailed_opportunities: str

def get_summary_data(brand_name: str) -> Dict:
    """
    Get the existing summary data for the brand
    """
    response = supabase.table('competitor_summary').select(
        'competitive_summary, gaps_opportunities'
    ).eq('brand_name', brand_name).execute()
    
    if not response.data:
        raise ValueError(f"No summary data found for {brand_name}")
    
    return response.data[0]

def analyze_opportunities(brand_name: str, summary_data: Dict) -> DetailedOpportunities:
    """
    Create detailed analysis of opportunities
    """
    prompt = f"""Based on this competitive analysis for {brand_name}:

    Market Overview: {summary_data.get('competitive_summary', 'N/A')}
    Initial Opportunities Identified: {summary_data.get('gaps_opportunities', 'N/A')}

    Take the existing opportunities identified, conduct additional research if needed and further reinforce the opportunites or existing gaps in the comeptitive loyalty landscape:
    1. Common gaps and opportunities
    2. Gaps and opportunities unique to the brand or industry

    Structure the response in bullet form. Ensure you elaborate on the gaps and opportunities and identify which could provide quick wins and which are longer-term strategic"""

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": f"You are an analyst helping {brand_name} find opportunities to differentiate it's future loyalty program."},
            {"role": "user", "content": prompt}
        ],
        response_format=DetailedOpportunities
    )
    
    return completion.choices[0].message.parsed

def update_opportunities_analysis(brand_name: str, analysis: DetailedOpportunities):
    """
    Update the summary table with detailed opportunities
    """
    try:
        response = supabase.table('competitor_summary').update({
            'detailed_opportunities': analysis.detailed_opportunities
        }).eq('brand_name', brand_name).execute()
        
        logger.info(f"Successfully updated detailed opportunities for {brand_name}")
        return response.data[0]
    except Exception as e:
        logger.error(f"Error updating analysis: {str(e)}")
        raise

# FastAPI endpoints
@app.get("/")
async def root():
    logger.info("Health check endpoint called")
    return {"status": "API is running"}

@app.post("/opportunities/{brand_name}", response_model=OpportunitiesResponse)
async def expand_opportunities_analysis(brand_name: str):
    """
    Create and save detailed opportunities analysis for a brand
    """
    try:
        logger.info(f"Starting opportunities analysis for brand: {brand_name}")
        
        # Get existing summary data
        summary_data = get_summary_data(brand_name)
        logger.info("Retrieved existing summary data")
        
        # Create detailed analysis
        detailed_analysis = analyze_opportunities(brand_name, summary_data)
        logger.info("Created detailed opportunities analysis")
        
        # Save the analysis
        updated_data = update_opportunities_analysis(brand_name, detailed_analysis)
        
        return OpportunitiesResponse(
            brand_name=brand_name,
            detailed_opportunities=detailed_analysis.detailed_opportunities
        )
        
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
