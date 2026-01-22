import logging
import json
import os
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    room_io,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.deepgram import STT as DeepgramSTT
from livekit.plugins.anthropic import LLM as AnthropicLLM
from livekit.plugins.cartesia import TTS as CartesiaTTS
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent import BookingAgent

load_dotenv()
logger = logging.getLogger("voice-agent")

# Import Beyond Presence plugin (optional)
try:
    from livekit.plugins import bey

    BEY_AVAILABLE = True
except ImportError:
    BEY_AVAILABLE = False
    logger.warning(
        "Beyond Presence plugin not available. Avatar functionality disabled."
    )

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Log room information (local participant not available until after connection)
    logger.info(f"Room: {ctx.room.name}")

    # Set up event listeners for participant tracking
    async def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(
            f"Participant connected - Identity: {participant.identity}, "
            f"Kind: {participant.kind}, SID: {participant.sid}"
        )
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            logger.info(f"✅ AGENT PARTICIPANT JOINED - Identity: {participant.identity}")
        # Log all tracks published by this participant
        for track_pub in participant.track_publications.values():
            logger.info(
                f"  Track: {track_pub.track_name} ({track_pub.kind}), "
                f"Source: {track_pub.source}, Muted: {track_pub.is_muted}"
            )

    async def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(
            f"Participant disconnected - Identity: {participant.identity}, "
            f"Kind: {participant.kind}"
        )
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
            logger.warning(f"⚠️ AGENT PARTICIPANT DISCONNECTED - Identity: {participant.identity}")

    async def on_track_published(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logger.info(
            f"Track published - Participant: {participant.identity} ({participant.kind}), "
            f"Track: {publication.track_name} ({publication.kind}), Source: {publication.source}"
        )

    async def on_track_unpublished(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logger.info(
            f"Track unpublished - Participant: {participant.identity} ({participant.kind}), "
            f"Track: {publication.track_name}"
        )

    # Register event listeners
    ctx.room.on("participant_connected", on_participant_connected)
    ctx.room.on("participant_disconnected", on_participant_disconnected)
    ctx.room.on("track_published", on_track_published)
    ctx.room.on("track_unpublished", on_track_unpublished)

    # Log existing remote participants
    if ctx.room.remote_participants:
        logger.info(f"Existing remote participants: {len(ctx.room.remote_participants)}")
        for participant in ctx.room.remote_participants.values():
            logger.info(
                f"  - Identity: {participant.identity}, Kind: {participant.kind}, "
                f"Tracks: {len(participant.track_publications)}"
            )
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                logger.info(f"  ✅ Found existing AGENT participant: {participant.identity}")

    # Initialize the booking agent
    # We pass the context to facilitate tool broadcasting if needed,
    # though strictly the Agent class doesn't require it in its constructor signature by default.
    # But since we defined BookingAgent, let's modify it to accept ctx or handle broadcasting via ctx here.
    # For now, let's keep BookingAgent compliant with the Base Agent signature (instructions, etc.)
    # and handle broadcasting by injecting ctx or a callback.

    agent = BookingAgent()
    # Inject ctx for broadcasting if we modify BookingAgent to use it
    agent.ctx = ctx

    # Implement the broadcast helper dynamically or rely on agent having 'ctx'
    async def _broadcast_tool_call(tool_name: str, args: dict):
        if ctx.room and ctx.room.isconnected():
            payload = json.dumps({"type": "tool_call", "tool": tool_name, "args": args})
            await ctx.room.local_participant.publish_data(payload.encode("utf-8"))

    # Monkey patch or bind the broadcast method?
    # Better: Update BookingAgent to use self.ctx if we set it.

    session = AgentSession(
        stt=DeepgramSTT(model="nova-2-general", language="en-US"),
        llm=AnthropicLLM(model="claude-sonnet-4-5-20250929"),
        tts=CartesiaTTS(
            model="sonic-2"
        ),  # default voice "Katie"; use voice="<id>" for custom
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Store references in agent so end_conversation can properly cleanup
    agent._agent_session = session
    agent._room = ctx.room

    # Initialize Beyond Presence avatar if API key is configured
    avatar = None
    bey_api_key = os.getenv("BEY_API_KEY")
    if BEY_AVAILABLE and bey_api_key:
        try:
            avatar_id = os.getenv(
                "BEY_AVATAR_ID", "b9be11b8-89fb-4227-8f86-4a881393cbdb"
            )
            logger.info(
                f"Initializing Beyond Presence avatar with ID: {avatar_id}/n {bey_api_key}"
            )
            avatar = bey.AvatarSession(avatar_id=avatar_id)
            # Start the avatar and wait for it to join
            await avatar.start(session, room=ctx.room)
            logger.info("Beyond Presence avatar started successfully")
            # Store avatar reference for cleanup
            agent._avatar = avatar
        except Exception as e:
            logger.error(f"Failed to start Beyond Presence avatar: {e}", exc_info=True)
            logger.warning("Continuing without avatar functionality")
            avatar = None
            agent._avatar = None
    elif BEY_AVAILABLE and not bey_api_key:
        logger.info("BEY_API_KEY not set. Avatar functionality disabled.")
        agent._avatar = None
    elif not BEY_AVAILABLE:
        logger.debug("Beyond Presence plugin not available")
        agent._avatar = None

    logger.info("Starting agent session...")
    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )
    logger.info("Agent session started successfully")

    logger.info("Connecting to room...")
    await ctx.connect()
    logger.info("Connected to room successfully")

    # Log final participant status after connection
    logger.info(f"Room state: {ctx.room.state}")
    if ctx.room.isconnected():
        logger.info(f"Local participant identity: {ctx.room.local_participant.identity}")
    logger.info(f"Remote participants count: {len(ctx.room.remote_participants)}")
    
    # Check for agent participants after connection
    agent_participants = [
        p for p in ctx.room.remote_participants.values()
        if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    ]
    if agent_participants:
        logger.info(f"✅ Agent participants found: {[p.identity for p in agent_participants]}")
    else:
        logger.warning("⚠️ No agent participants found in room after connection")


if __name__ == "__main__":
    cli.run_app(server)
