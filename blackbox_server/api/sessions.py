from __future__ import annotations

from fastapi import APIRouter, Query, Request

from blackbox_server.core.models import (
    AbortResponse,
    ExecuteCmdRequest,
    ExecuteCmdResponse,
    MessageRequest,
    MessageResponse,
    SessionResponse,
)


router = APIRouter()


@router.post("/v1/sessions/{session_id}/messages", response_model=MessageResponse)
async def send_message(session_id: str, request: Request, payload: MessageRequest) -> MessageResponse:
    server = request.app.state.server
    async with server.request_scope():
        response = await server.send_message(session_id, payload)
    response.request_id = request.state.request_id
    return response


@router.post("/v1/sessions/{session_id}/execute_cmd", response_model=ExecuteCmdResponse)
async def execute_cmd(session_id: str, request: Request, payload: ExecuteCmdRequest) -> ExecuteCmdResponse:
    server = request.app.state.server
    async with server.request_scope():
        response = await server.execute_cmd(session_id, payload)
    response.request_id = request.state.request_id
    return response


@router.get("/v1/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    request: Request,
    include_history: bool = Query(default=False),
    include_trace: bool = Query(default=False),
    include_turns: bool = Query(default=False),
) -> SessionResponse:
    server = request.app.state.server
    async with server.request_scope():
        response = await server.get_session(
            session_id,
            include_history=include_history,
            include_trace=include_trace,
            include_turns=include_turns,
        )
    response.request_id = request.state.request_id
    return response


@router.post("/v1/sessions/{session_id}/abort", response_model=AbortResponse)
async def abort_session(session_id: str, request: Request) -> AbortResponse:
    server = request.app.state.server
    async with server.request_scope():
        response = await server.abort_session(session_id)
    response.request_id = request.state.request_id
    return response
