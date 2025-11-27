from livekit import api
from livekit.protocol import sip as sip_protocol
from logging import getLogger
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv("./.env", override=True)

class Settings(BaseSettings):
    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str
    LIVEKIT_URL: str

    TWILIO_SIP_DOMAIN: str
    TWILIO_PHONE_NUMBER: str
    TWILIO_SIP_USERNAME: str
    TWILIO_SIP_PASSWORD: str

settings = Settings()


# Initialize logger
logger = getLogger(__name__)
logger.setLevel("INFO")


async def create_outbound_sip_trunk_if_not_exists() -> str:
    """Create or get existing SIP trunk for outbound calls."""
    
    if not all([settings.TWILIO_SIP_DOMAIN, settings.TWILIO_PHONE_NUMBER, 
                settings.TWILIO_SIP_USERNAME, settings.TWILIO_SIP_PASSWORD]):
        raise ValueError("Twilio SIP credentials not configured")
    
    livekit_api = api.LiveKitAPI(
        api_key=settings.LIVEKIT_API_KEY,
        api_secret=settings.LIVEKIT_API_SECRET,
        url=settings.LIVEKIT_URL,
    )

    try:
        # List existing trunks
        list_request = sip_protocol.ListSIPOutboundTrunkRequest()
        existing_trunks = await livekit_api.sip.list_sip_outbound_trunk(list=list_request)
        
        # Check if trunk already exists
        for trunk in existing_trunks.items:
            if trunk.name == "FinancialAssistantTrunk":
                logger.info(f"Using existing SIP trunk: {trunk.sip_trunk_id}")
                return trunk.sip_trunk_id
            
        logger.info("Creating new SIP Outbound Trunk...")
        new_trunk_info = sip_protocol.SIPOutboundTrunkInfo(
            name="FinancialAssistantTrunk",
            address=settings.TWILIO_SIP_DOMAIN,
            numbers=[settings.TWILIO_PHONE_NUMBER],
            auth_password=settings.TWILIO_SIP_PASSWORD,
            auth_username=settings.TWILIO_SIP_USERNAME,
        )

        create = sip_protocol.CreateSIPOutboundTrunkRequest(trunk=new_trunk_info)
        trunk = await livekit_api.sip.create_sip_outbound_trunk(create)
        
        logger.info(f"Created new SIP Trunk: {trunk.sip_trunk_id}")
        return trunk.sip_trunk_id
        
    finally:
        await livekit_api.aclose()


async def make_call(phone: str):
    """Initiate outbound call using Twilio SIP."""
    
    livekit_api = api.LiveKitAPI(
        api_key=settings.LIVEKIT_API_KEY,
        api_secret=settings.LIVEKIT_API_SECRET,
        url=settings.LIVEKIT_URL
    )

    trunk_id = await create_outbound_sip_trunk_if_not_exists()

    try:
        create = sip_protocol.CreateSIPParticipantRequest(
            sip_trunk_id=trunk_id,
            sip_call_to=phone,
            room_name=f"financial_assistant:{phone}",
            participant_identity=phone,
            participant_name="User",
            krisp_enabled=True,
            wait_until_answered=True,
        )

        logger.info(f"Calling {phone}...")
        await livekit_api.sip.create_sip_participant(create=create)
        logger.info(f"Call initiated to {phone}")

    except Exception as e:
        logger.error(f"Call failed: {e}")
    finally:
        await livekit_api.aclose()

if __name__ == "__main__":
    import asyncio
    phone_number = "+919123561817"  # Replace with the target phone number
    asyncio.run(make_call(phone_number))