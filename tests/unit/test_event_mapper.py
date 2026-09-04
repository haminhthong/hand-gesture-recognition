"""Unit tests cho mô-đun event_mapper."""

from hand_gesture_controller.event_mapper import GestureEvent, GestureEventMapper


def test_event_mapper_motion_gestures():
    mapper = GestureEventMapper()
    assert mapper.map_gesture_to_event("Select", "On/Off") == GestureEvent.TOGGLE_CANVAS
    assert mapper.map_gesture_to_event("Stop", "SOS") == GestureEvent.EMERGENCY_SOS


def test_event_mapper_static_drag_sequence():
    mapper = GestureEventMapper()

    # Initial Select -> START_DRAG
    evt1 = mapper.map_gesture_to_event("Select", "Still")
    assert evt1 == GestureEvent.START_DRAG
    assert mapper.is_dragging is True

    # Subsequent Select -> DRAG
    evt2 = mapper.map_gesture_to_event("Select", "Still")
    assert evt2 == GestureEvent.DRAG
    assert mapper.is_dragging is True

    # Transition away from Select -> STOP_DRAG
    evt3 = mapper.map_gesture_to_event("Fist", "Still")
    assert evt3 == GestureEvent.STOP_DRAG
    assert mapper.is_dragging is False


def test_event_mapper_actions():
    mapper = GestureEventMapper()
    assert mapper.map_gesture_to_event("Options", "Still") == GestureEvent.CHANGE_COLOR
    assert mapper.map_gesture_to_event("Stop", "Still") == GestureEvent.DELETE_OBJECT
    assert mapper.map_gesture_to_event("Peace", "Still") == GestureEvent.OPEN_MENU
    assert mapper.map_gesture_to_event("Fist", "Still") == GestureEvent.NONE
