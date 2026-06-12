# Hermes Voice v02 LiveKit Build

Status: staged for WebRTC MVP before phone-number signup.

## Current Path

Voice v02 uses LiveKit first:

1. WebRTC room token smoke without a phone number.
2. LiveKit agent dispatch into a unique room.
3. SIP inbound trunk after Pafi buys a number.
4. Same Hermes voice telemetry shape: STT, brain, TTS, send, total.

Telegram live calls stay experimental and separate.

## Environment

Required for WebRTC MVP:

```bash
export LIVEKIT_URL='wss://...'
export LIVEKIT_API_KEY='...'
export LIVEKIT_API_SECRET='...'
export HERMES_LIVEKIT_AGENT_NAME='hermes-live-voice'
export HERMES_LIVEKIT_ROOM_PREFIX='hermes-call-'
```

Required after buying a number:

```bash
export HERMES_LIVEKIT_PHONE_NUMBER='+407...'
export HERMES_LIVEKIT_SIP_PROVIDER='telnyx'
```

Required before outbound PSTN calls:

```bash
# Preferred: create one stored outbound trunk with Twilio, Telnyx, or another SIP provider.
export HERMES_LIVEKIT_OUTBOUND_TRUNK_ID='ST_xxx'

# Optional inline trunk mode for smoke setup only.
export HERMES_LIVEKIT_OUTBOUND_SIP_ADDRESS='sip.telnyx.com'
export HERMES_LIVEKIT_OUTBOUND_SIP_AUTH_USERNAME='...'
export HERMES_LIVEKIT_OUTBOUND_SIP_AUTH_PASSWORD='...'
export HERMES_LIVEKIT_OUTBOUND_FROM_NUMBER='+15105550100'
export HERMES_LIVEKIT_OUTBOUND_DESTINATION_COUNTRY='US'
```

LiveKit-managed phone numbers are inbound-only today. Do not assume
`HERMES_LIVEKIT_PHONE_NUMBER` can be used as outbound caller ID unless the SIP
provider explicitly owns or verifies that number.

## Commands

Preflight without requiring a phone number:

```bash
python -m gateway.livekit_voice preflight
```

Preflight once SIP is expected:

```bash
python -m gateway.livekit_voice preflight --require-phone-number
```

Generate a WebRTC token for browser MVP:

```bash
python -m gateway.livekit_voice room-token --identity pafi
```

Generate explicit agent dispatch JSON:

```bash
python -m gateway.livekit_voice dispatch-json --trunk-id ST_xxx
```

Generate inbound trunk JSON after buying the number:

```bash
python -m gateway.livekit_voice inbound-trunk-json --phone-number +40740000000
```

Generate outbound trunk JSON for a SIP provider number:

```bash
python -m gateway.livekit_voice outbound-trunk-json \
  --address sip.telnyx.com \
  --number +15105550100
```

Create a dry-run outbound call plan for the PA or concierge profile:

```bash
python -m gateway.livekit_voice outbound-call-json \
  --profile pa \
  --to-number +12135550100 \
  --purpose "Confirm appointment time" \
  --trunk-id ST_xxx
```

Execute an outbound call only after the stored trunk or inline SIP credentials
are configured:

```bash
python -m gateway.livekit_voice outbound-call \
  --profile concierge \
  --to-number +12135550100 \
  --purpose "Restaurant booking callback" \
  --execute
```

## Acceptance Gates

- Preflight redacts all LiveKit secrets.
- WebRTC MVP is allowed to pass without a phone number.
- SIP preflight fails until `HERMES_LIVEKIT_PHONE_NUMBER` is set.
- Dispatch rule uses explicit `agentName`, not automatic dispatch.
- Inbound trunk JSON accepts only +E.164 phone numbers.
- Outbound dry-run accepts only `pa` or `concierge` profiles.
- Outbound calls require a purpose and +E.164 callee number.
- Outbound execution fails closed unless a stored outbound trunk or complete
  inline SIP trunk config exists.
