import pytest
from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    AllocationLaunch,
    AllocationLaunchLocal,
    AuxiliaryModelInput,
    AuxiliaryModelKind,
    AuxiliaryModelLocator,
    Parallelism,
    ServeModelInput,
    ServeProcessAllocationModelRank,
    ServeRoleKind,
    resolve_draft_model_locator,
)

DRAFT_DECLARATION = AuxiliaryModelInput(
    kind=AuxiliaryModelKind(),
    model=ServeModelInput(id="deepseek-v4-flash-draft", served_name="deepseek-v4-flash-draft"),
)


def _locator(locator: str) -> AuxiliaryModelLocator:
    return AuxiliaryModelLocator(kind=AuxiliaryModelKind(), locator=locator)


def _allocation(locators: list[AuxiliaryModelLocator] | None) -> ServeProcessAllocationModelRank:
    return ServeProcessAllocationModelRank(
        process="serve",
        role="serve",
        role_kind=ServeRoleKind.serve,
        replica=0,
        rank=0,
        rank_count=1,
        machine="node-a",
        model_locator="/models/deepseek-v4-flash",
        cache="/cache/runtime/node-a/serve",
        devices=[],
        launch=AllocationLaunch(root=AllocationLaunchLocal()),
        effective_settings={},
        effective_parallelism=Parallelism(),
        ports={},
        auxiliary_model_locators=locators,
    )


def test_resolve_draft_model_locator_returns_the_carried_locator() -> None:
    allocation = _allocation([_locator("/models/deepseek-v4-flash-draft")])

    assert (
        resolve_draft_model_locator(allocation, [DRAFT_DECLARATION])
        == "/models/deepseek-v4-flash-draft"
    )


def test_resolve_draft_model_locator_is_none_without_declaration_or_locator() -> None:
    assert resolve_draft_model_locator(_allocation(None), None) is None
    assert resolve_draft_model_locator(_allocation([]), []) is None


def test_resolve_draft_model_locator_returns_an_unmatched_carried_locator() -> None:
    # A carried locator is spliced even without a matching declaration; the
    # cardinality rule applies either way.
    allocation = _allocation([_locator("/models/deepseek-v4-flash-draft")])

    assert resolve_draft_model_locator(allocation, None) == "/models/deepseek-v4-flash-draft"


def test_resolve_draft_model_locator_rejects_a_declaration_without_a_locator() -> None:
    with pytest.raises(AdapterOperationError, match="carries no draft-model locator") as captured:
        resolve_draft_model_locator(_allocation(None), [DRAFT_DECLARATION])

    assert captured.value.code == AdapterErrorCode.invalid_request


def test_resolve_draft_model_locator_rejects_multiple_locators() -> None:
    allocation = _allocation(
        [_locator("/models/deepseek-v4-flash-draft"), _locator("/models/other-draft")]
    )

    with pytest.raises(AdapterOperationError, match="expected exactly one") as captured:
        resolve_draft_model_locator(allocation, [DRAFT_DECLARATION])

    assert captured.value.code == AdapterErrorCode.invalid_request
