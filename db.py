import os
import logging
from supabase import create_client, Client

logger = logging.getLogger("voice-agent")


class Database:
    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            logger.warning(
                "Supabase credentials not found. Database operations will fail."
            )
            self.client = None
        else:
            self.client: Client = create_client(url, key)

    def get_appointments(self, user_contact: str):
        if not self.client:
            return []
        try:
            response = (
                self.client.table("appointments")
                .select("*")
                .eq("contact_number", user_contact)
                .execute()
            )
            return response.data
        except Exception as e:
            logger.error(f"Error fetching appointments: {e}")
            return []

    def create_appointment(
        self, user_contact: str, user_name: str, slot: str, details: str
    ):
        if not self.client:
            return None
        try:
            data = {
                "contact_number": user_contact,
                "user_name": user_name,
                "slot_time": slot,
                "details": details,
                "status": "confirmed",
            }
            response = self.client.table("appointments").insert(data).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error creating appointment: {e}")
            return None

    def cancel_appointment(self, appointment_id: str):
        if not self.client:
            return None
        try:
            response = (
                self.client.table("appointments")
                .update({"status": "cancelled"})
                .eq("id", appointment_id)
                .execute()
            )
            return response.data
        except Exception as e:
            logger.error(f"Error cancelling appointment: {e}")
            return None

    def update_appointment(self, appointment_id: str, new_slot: str):
        if not self.client:
            return None
        try:
            response = (
                self.client.table("appointments")
                .update({"slot_time": new_slot})
                .eq("id", appointment_id)
                .execute()
            )
            return response.data
        except Exception as e:
            logger.error(f"Error updating appointment: {e}")
            return None

    def is_slot_available(self, slot: str):
        """Check if a slot is available (not booked by any customer).

        Args:
            slot: The date and time of the slot to check.

        Returns:
            True if the slot is available, False if it's already booked.
        """
        if not self.client:
            return True  # If no DB, assume available
        try:
            # Check if any confirmed appointment exists for this slot
            response = (
                self.client.table("appointments")
                .select("id")
                .eq("slot_time", slot)
                .eq("status", "confirmed")
                .execute()
            )
            is_available = len(response.data) == 0
            logger.info(
                f"Slot {slot} availability check: {is_available} (found {len(response.data)} existing bookings)"
            )
            return is_available
        except Exception as e:
            logger.error(f"Error checking slot availability: {e}")
            # On error, assume available to avoid blocking legitimate bookings
            return True
