"""api/routers/game.py — Game model, Bayesian belief, and Nash solver endpoints."""

from fastapi import APIRouter, Depends
from api.dependencies import (
    get_belief_updater, get_nash_solver,
    get_attacker_model, get_game, get_game_metrics,
)
from api.schemas.game import BeliefState, AttackerPrediction, GameState

router = APIRouter()


@router.get("/belief", response_model=BeliefState, summary="Get current Bayesian belief")
async def get_belief(updater=Depends(get_belief_updater)):
    """Return the current posterior belief over attacker type."""
    if updater is None:
        return BeliefState()
    belief = updater.get_current_belief()
    return BeliefState(**belief.to_dict())


@router.get("/belief/history", summary="Belief history over time")
async def get_belief_history(
    updater = Depends(get_belief_updater),
    metrics = Depends(get_game_metrics),
):
    """Return the belief probability history for the dashboard chart."""
    if metrics is None:
        return {"history": []}
    return {"history": metrics.get_belief_history(last_n=500)}


@router.get("/predictions", response_model=AttackerPrediction, summary="Predicted next actions")
async def get_attacker_predictions(
    updater        = Depends(get_belief_updater),
    attacker_model = Depends(get_attacker_model),
    game           = Depends(get_game),
):
    """Return predicted attacker next actions given current belief."""
    if updater is None or attacker_model is None or game is None:
        return AttackerPrediction()

    belief = updater.get_current_belief()
    state  = game.state

    if state is None:
        return AttackerPrediction(
            dominant_type        = belief.dominant_type,
            dominant_probability = belief.dominant_probability,
            recommendation       = updater.get_recommended_strategy(),
        )

    strategies = attacker_model.get_all_strategies(state)
    return AttackerPrediction(
        dominant_type        = belief.dominant_type,
        dominant_probability = belief.dominant_probability,
        probabilities        = belief.probabilities,
        strategies           = strategies,
        recommendation       = updater.get_recommended_strategy(),
    )


@router.get("/state", response_model=GameState, summary="Current game state")
async def get_game_state(game=Depends(get_game)):
    """Return the current stochastic game state."""
    if game is None or game.state is None:
        return GameState()
    return GameState(**game.state.to_dict())


@router.get("/nash", summary="Nash equilibrium and action recommendations")
async def get_nash_recommendations(
    game           = Depends(get_game),
    updater        = Depends(get_belief_updater),
    attacker_model = Depends(get_attacker_model),
    nash_solver    = Depends(get_nash_solver),
):
    """Return Nash equilibrium mixed strategy and top-k recommended actions."""
    if any(x is None for x in [game, updater, attacker_model, nash_solver]):
        return {"available": False}
    if game.state is None:
        return {"available": False}

    belief = updater.get_current_belief()
    return {
        "available":      True,
        "recommendations": nash_solver.get_action_recommendation(
            state          = game.state,
            attacker_model = attacker_model,
            belief         = belief,
            top_k          = 5,
        ),
    }


@router.get("/metrics", summary="Game-theoretic metrics")
async def get_game_metrics_endpoint(metrics=Depends(get_game_metrics)):
    """Return the current game metrics snapshot."""
    if metrics is None:
        return {}
    return metrics.get_snapshot().to_dict()


@router.get("/metrics/paper", summary="Paper evaluation metrics")
async def get_paper_metrics(metrics=Depends(get_game_metrics)):
    """Return metrics formatted for the paper's game model evaluation table."""
    if metrics is None:
        return {}
    return metrics.get_paper_metrics()