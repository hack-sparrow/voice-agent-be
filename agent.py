import logging
import json
import asyncio
from typing import Annotated
from livekit.agents import (
    Agent,
    llm,
)
from livekit.agents.llm import ChatContext
from db import Database

logger = logging.getLogger("agent")


class BookingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant for booking appointments. "
            "Your goal is to help users book, modify, or cancel appointments. "
            "CRITICAL: At the very start of the conversation, you MUST immediately ask for and collect the user's contact number (phone number) using the identify_user tool. "
            "Do not proceed with any other actions until you have the user's contact number. "
            "Always be polite, concise, and confirm details before booking. "
            "If you need to perform an action, use the appropriate tool. "
            "When the user says goodbye, thanks you, says 'that's it', 'that's all', or indicates they are done, "
            "you MUST call the end_conversation tool to properly end the call and disconnect.",
        )
        self.db = Database()
        self.user_contact = None
        self.user_name = None
        self._agent_session = (
            None  # Will be set by main.py to store AgentSession reference
        )
        self._room = None  # Will be set by main.py to store Room reference
        self._avatar = None  # Will be set by main.py to store AvatarSession reference

    async def _broadcast_tool_call(self, tool_name: str, args: dict):
        """Helper to send tool execution status to frontend"""
        try:
            if (
                hasattr(self, "ctx")
                and self.ctx
                and self.ctx.room
                and self.ctx.room.isconnected()
            ):
                payload = json.dumps(
                    {"type": "tool_call", "tool": tool_name, "args": args}
                )
                await self.ctx.room.local_participant.publish_data(
                    payload.encode("utf-8")
                )
        except Exception as e:
            logger.error(f"Failed to broadcast tool call: {e}")

    @llm.function_tool
    async def identify_user(self, contact_number: str):
        """Identify the user by their phone number.

        This MUST be called at the very start of every conversation before any other actions.
        Ask the user for their contact/phone number and use this tool to store it.

        Args:
            contact_number: The user's contact/phone number.
        """
        await self._broadcast_tool_call(
            "identify_user", {"contact_number": contact_number}
        )
        logger.info(f"Identifying user: {contact_number}")
        self.user_contact = contact_number
        return f"Thank you! I have your contact number {contact_number}. How can I help you today?"

    @llm.function_tool
    async def fetch_slots(self):
        """Fetch available slots for appointments."""
        await self._broadcast_tool_call("fetch_slots", {})
        logger.info("Fetching slots")
        return (
            "Available slots are: 10:30am - 11:30am, 26th January;"
            " 2:15pm - 3:15pm, 26th January; "
            "9:00am - 10:00am, 27th January; "
            "3:45pm - 4:45pm, 27th January; "
            "11:00am - 12:00pm, 28th January; "
            "1:30pm - 2:30pm, 28th January; "
            "10:00am - 11:00am, 29th January; "
            "4:00pm - 5:00pm, 29th January; "
            "9:15am - 10:15am, 30th January; "
            "2:00pm - 3:00pm, 30th January."
        )

    @llm.function_tool
    async def book_appointment(self, slot: str, details: str):
        """Book an appointment for the user.

        Args:
            slot: The date and time of the slot to book.
            details: Purpose or details of the appointment.
        """
        await self._broadcast_tool_call(
            "book_appointment", {"slot": slot, "details": details}
        )
        logger.info(f"Booking appointment: {slot} - {details}")

        if not self.user_contact:
            return "I need your contact number before booking. Please provide it using the identify_user tool."

        # Check if this user already has an appointment at this slot
        existing = self.db.get_appointments(self.user_contact)
        for appt in existing:
            if appt.get("slot_time") == slot and appt.get("status") == "confirmed":
                return f"You already have an appointment at {slot}."

        # Check if ANY customer has already booked this slot (prevent double booking)
        slot_available = self.db.is_slot_available(slot)
        if not slot_available:
            return f"I'm sorry, but the slot {slot} is already booked by another customer. Please choose a different time slot."

        result = self.db.create_appointment(
            self.user_contact, self.user_name or "Unknown", slot, details
        )
        if result:
            return f"Appointment booked successfully for {slot}. Your appointment details: {details}."
        else:
            return "I'm sorry, I couldn't book the appointment due to a system error."

    @llm.function_tool
    async def retrieve_appointments(self):
        """Retrieve user's past and upcoming appointments."""
        await self._broadcast_tool_call("retrieve_appointments", {})
        if not self.user_contact:
            return "Please provide your contact number first."

        appts = self.db.get_appointments(self.user_contact)
        if not appts:
            return "You have no appointments on record."

        appt_list = [
            f"{a.get('slot_time')} ({a.get('details')}) - {a.get('status')}"
            for a in appts
        ]
        return "Here are your appointments: " + ", ".join(appt_list)

    @llm.function_tool
    async def cancel_appointment(self, slot: str):
        """Cancel an appointment.

        Args:
            slot: The slot time of the appointment to cancel.
        """
        await self._broadcast_tool_call("cancel_appointment", {"slot": slot})
        if not self.user_contact:
            return "Please provide your phone number."

        appts = self.db.get_appointments(self.user_contact)
        target = next(
            (
                a
                for a in appts
                if a.get("slot_time") == slot and a.get("status") != "cancelled"
            ),
            None,
        )

        if target:
            self.db.cancel_appointment(target["id"])
            return f"Appointment at {slot} has been cancelled."
        else:
            return f"I couldn't find an active appointment at {slot}."

    @llm.function_tool
    async def modify_appointment(self, old_slot: str, new_slot: str):
        """Modify an existing appointment.

        Args:
            old_slot: The current slot time.
            new_slot: The new desired slot time.
        """
        await self._broadcast_tool_call(
            "modify_appointment", {"old_slot": old_slot, "new_slot": new_slot}
        )
        if not self.user_contact:
            return "Please identify yourself first using the identify_user tool."

        appts = self.db.get_appointments(self.user_contact)
        target = next(
            (
                a
                for a in appts
                if a.get("slot_time") == old_slot and a.get("status") != "cancelled"
            ),
            None,
        )

        if not target:
            return f"Could not find appointment at {old_slot}."

        # Check if the new slot is available (prevent double booking)
        slot_available = self.db.is_slot_available(new_slot)
        if not slot_available:
            return f"I'm sorry, but the slot {new_slot} is already booked by another customer. Please choose a different time slot."

        self.db.update_appointment(target["id"], new_slot)
        return f"Appointment changed from {old_slot} to {new_slot}."

    @llm.function_tool
    async def end_conversation(self):
        """Ends the conversation and disconnects the call.

        Call this tool when:
        - The user says goodbye, thanks you, or indicates they are done
        - The user says phrases like 'that's it', 'that's all', 'I'm done', 'thank you', 'bye'
        - The conversation is complete and the user wants to end the call

        This will properly disconnect the call. Always respond with a friendly goodbye message
        before calling this tool, or return the goodbye message from this tool.
        """
        await self._broadcast_tool_call("end_conversation", {})

        summary = None

        # Schedule disconnection to happen after the goodbye message is spoken
        async def _disconnect_after_goodbye():
            """Disconnect after the goodbye message audio finishes playing"""
            try:
                # Wait for the goodbye message to be generated, converted to audio, and played
                # The LLM needs to process the tool return, TTS needs to generate audio, and it needs to play
                # A reasonable estimate: 5-8 seconds should be enough for most goodbye messages
                goodbye_wait_time = 8.0
                logger.info(
                    f"Waiting {goodbye_wait_time}s for goodbye message to be spoken..."
                )

                # Wait a bit for the message to start being processed
                await asyncio.sleep(1.0)

                # Then try to wait for audio to finish, with a timeout
                if self._agent_session and self._agent_session.output.audio:
                    try:
                        # Flush to ensure any pending audio is sent
                        self._agent_session.output.audio.flush()
                        # Wait for audio with timeout
                        await asyncio.wait_for(
                            self._agent_session.output.audio.wait_for_playout(),
                            timeout=goodbye_wait_time - 1.0,
                        )
                        logger.info("Goodbye message audio finished")
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Timeout waiting for audio, proceeding with disconnect"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error waiting for audio playback: {e}", exc_info=True
                        )
                        # Wait the remaining time as fallback
                        remaining_time = goodbye_wait_time - 1.0
                        if remaining_time > 0:
                            await asyncio.sleep(remaining_time)
                else:
                    # If no audio output, wait the full time
                    await asyncio.sleep(goodbye_wait_time - 1.0)

                # Now perform the actual disconnection
                logger.info("Proceeding with disconnection...")
                await self._perform_disconnection()
            except Exception as e:
                logger.error(f"Error in disconnect task: {e}", exc_info=True)
                # Try to disconnect anyway
                try:
                    await self._perform_disconnection()
                except Exception as cleanup_error:
                    logger.error(
                        f"Error during forced cleanup: {cleanup_error}", exc_info=True
                    )

        # Start the disconnection task in the background
        # Don't await it - let it run in the background so the tool can return immediately
        if self._agent_session:
            asyncio.create_task(_disconnect_after_goodbye())

        try:
            # Step 1: Wait for any current audio to finish (before goodbye message)
            if self._agent_session and self._agent_session.output.audio:
                logger.info("Waiting for current audio playback to finish...")
                try:
                    # Flush any remaining audio to ensure it's sent
                    self._agent_session.output.audio.flush()
                    # Wait for current audio segments to finish playing
                    await self._agent_session.output.audio.wait_for_playout()
                    logger.info("Current audio playback finished")
                except Exception as e:
                    logger.warning(
                        f"Error waiting for audio playback: {e}", exc_info=True
                    )
                    # Continue even if waiting for audio fails

            # Step 2: Generate conversation summary using LLM
            if self._agent_session and self._agent_session.llm:
                logger.info("Generating conversation summary...")
                try:
                    # Collect conversation messages for summarization
                    conversation_messages = []
                    for item in self.chat_ctx.items:
                        if isinstance(item, llm.ChatMessage) and item.text_content:
                            # Skip system messages and summaries
                            if item.role not in ("user", "assistant"):
                                continue
                            if item.extra and item.extra.get("is_summary"):
                                continue
                            conversation_messages.append(
                                f"{item.role}: {item.text_content.strip()}"
                            )

                    if conversation_messages:
                        # Create a summary context with the conversation
                        summary_ctx = ChatContext()
                        summary_ctx.add_message(
                            role="system",
                            content=(
                                "Compress the conversation into a short, faithful summary.\n"
                                "Focus on user goals, constraints, decisions, key facts/preferences/entities, and pending tasks.\n"
                                "Exclude chit-chat and greetings. Be concise. Include any appointments that were booked, modified, or cancelled."
                            ),
                        )
                        summary_ctx.add_message(
                            role="user",
                            content=f"Conversation to summarize:\n\n"
                            + "\n".join(conversation_messages),
                        )

                        # Generate summary using LLM
                        chunks = []
                        async for chunk in self._agent_session.llm.chat(
                            chat_ctx=summary_ctx
                        ):
                            if chunk.delta and chunk.delta.content:
                                chunks.append(chunk.delta.content)

                        summary = "".join(chunks).strip()
                        if summary:
                            logger.info(f"Generated summary: {summary[:200]}...")
                        else:
                            # Fallback: use conversation messages
                            summary = "\n".join(
                                conversation_messages[-10:]
                            )  # Last 10 messages
                            logger.info("Using fallback summary from chat history")
                    else:
                        summary = "No conversation history available."
                        logger.info("No conversation messages to summarize")
                except Exception as e:
                    logger.error(f"Error generating summary: {e}", exc_info=True)
                    # Fallback to simple history extraction
                    history_parts = []
                    for item in self.chat_ctx.items:
                        if isinstance(item, llm.ChatMessage) and item.text_content:
                            history_parts.append(f"{item.role}: {item.text_content}")
                    summary = (
                        "\n".join(history_parts)
                        if history_parts
                        else "No conversation history available."
                    )
            else:
                # Fallback: extract history if LLM is not available
                history_parts = []
                for item in self.chat_ctx.items:
                    if isinstance(item, llm.ChatMessage) and item.text_content:
                        history_parts.append(f"{item.role}: {item.text_content}")
                summary = (
                    "\n".join(history_parts)
                    if history_parts
                    else "No conversation history available."
                )

            # Log the final summary
            logger.info(f"Conversation summary: {summary[:500]}...")

            # Get booked appointments for the summary
            booked_appointments = []
            if self.user_contact:
                try:
                    appts = self.db.get_appointments(self.user_contact)
                    booked_appointments = [
                        f"{a.get('slot_time')} - {a.get('details')} ({a.get('status')})"
                        for a in appts
                        if a.get("status") == "confirmed"
                    ]
                    if booked_appointments:
                        logger.info(
                            f"User has {len(booked_appointments)} confirmed appointments"
                        )
                except Exception as e:
                    logger.warning(f"Error retrieving appointments for summary: {e}")

            # Broadcast summary to frontend if room is available and connected
            if self._room and self._room.isconnected():
                try:
                    summary_payload = json.dumps(
                        {
                            "type": "conversation_summary",
                            "summary": summary,
                            "appointments": booked_appointments,
                            "user_contact": self.user_contact,
                        }
                    )
                    await self._room.local_participant.publish_data(
                        summary_payload.encode("utf-8")
                    )
                    logger.info("Summary broadcasted to frontend")
                except Exception as e:
                    logger.warning(f"Failed to broadcast summary: {e}")

        except Exception as e:
            logger.error(f"Error during summary generation: {e}", exc_info=True)
            # Continue even if summary generation fails

        # Return the goodbye message immediately so it can be spoken
        # The disconnection will happen in the background task after the audio finishes
        return "Thank you for calling. Have a great day! Goodbye!"

    async def _perform_disconnection(self):
        """Perform the actual disconnection and cleanup"""
        try:
            # 1. Shutdown the agent session (this will stop audio/video streams)
            if self._agent_session:
                logger.info("Shutting down agent session...")
                # Use drain=True to allow current speech to finish before shutting down
                self._agent_session.shutdown(drain=True)
                # Wait for the shutdown to complete
                if (
                    hasattr(self._agent_session, "_closing_task")
                    and self._agent_session._closing_task
                ):
                    try:
                        await asyncio.wait_for(
                            self._agent_session._closing_task, timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Session shutdown timed out")
                    except Exception as e:
                        logger.warning(f"Error waiting for session shutdown: {e}")
            else:
                logger.warning("Session not available, cannot shutdown")

            # 2. Stop all published tracks (audio/video) from local participant
            if self._room and self._room.isconnected() and self._room.local_participant:
                logger.info("Stopping all published tracks...")
                for track_pub in list(
                    self._room.local_participant.track_publications.values()
                ):
                    if track_pub.track:
                        try:
                            await track_pub.track.stop()
                            logger.debug(f"Stopped track: {track_pub.track.kind}")
                        except Exception as e:
                            logger.warning(
                                f"Error stopping track {track_pub.track.kind}: {e}"
                            )

            # 3. Disconnect from the room (this will drop the agent from the room)
            if self._room and self._room.isconnected():
                logger.info("Disconnecting from room...")
                await self._room.disconnect()
                logger.info("Disconnected from room successfully")
            else:
                logger.debug("Room not connected or not available")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
