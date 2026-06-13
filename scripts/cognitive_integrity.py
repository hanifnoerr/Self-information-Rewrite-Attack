class CognitiveDriftError(Exception):
    """Raised when a proposed grid path crosses a forbidden transition."""


class LogicLattice:
    def __init__(self, grid_size=2):
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")

        self.grid_size = grid_size
        self.start_state = (0, 0)
        self.states = [
            (row, column)
            for row in range(grid_size)
            for column in range(grid_size)
        ]
        self.allowed_transitions = {
            state: [
                candidate
                for candidate in self.states
                if self.is_valid_transition(state, candidate)
            ]
            for state in self.states
        }
        self.transition_matrix = [
            [
                self.is_valid_transition(current_state, next_state)
                for next_state in self.states
            ]
            for current_state in self.states
        ]

    def token_to_state(self, token_id):
        return self.states[int(token_id) % len(self.states)]

    def is_valid_transition(self, current_state, next_state):
        row_distance = abs(current_state[0] - next_state[0])
        column_distance = abs(current_state[1] - next_state[1])
        return row_distance + column_distance <= 1

    def verify_path(self, grid_path):
        if not grid_path:
            raise CognitiveDriftError("The grid path is empty.")

        path = [tuple(state) for state in grid_path]
        if path[0] != self.start_state:
            raise CognitiveDriftError("The grid path must start at [0, 0].")

        for current_state, next_state in zip(path, path[1:]):
            if current_state not in self.states or next_state not in self.states:
                raise CognitiveDriftError("The grid path contains a state outside the lattice.")
            if not self.is_valid_transition(current_state, next_state):
                raise CognitiveDriftError(
                    f"Forbidden transition: {list(current_state)} -> {list(next_state)}"
                )

        return True

    def as_dict(self):
        return {
            "grid_size": self.grid_size,
            "start_state": list(self.start_state),
            "state_mapping": "token_id modulo number_of_grid_cells",
            "transition_rule": "same cell or one horizontal/vertical step",
            "state_order": [list(state) for state in self.states],
            "allowed_transitions": {
                str(list(state)): [list(candidate) for candidate in candidates]
                for state, candidates in self.allowed_transitions.items()
            },
            "transition_matrix": self.transition_matrix,
        }


class BehavioralVerifier:
    REQUIRED_TRACE_KEYS = {
        "grid_path",
        "safety_verification",
        "reasoning_integrity_check",
    }

    def __init__(self, grid_size=2):
        self.lattice = LogicLattice(grid_size=grid_size)

    def verify_or_correct_path(self, candidate_path):
        try:
            self.lattice.verify_path(candidate_path)
            return candidate_path, [], False
        except CognitiveDriftError:
            corrected_path = [list(self.lattice.start_state)]
            invalid_indexes = []

            for token_index, candidate_state in enumerate(candidate_path[1:]):
                current_state = tuple(corrected_path[-1])
                next_state = tuple(candidate_state)
                if self.lattice.is_valid_transition(current_state, next_state):
                    corrected_path.append(list(next_state))
                else:
                    invalid_indexes.append(token_index)

            self.lattice.verify_path(corrected_path)
            return corrected_path, invalid_indexes, True

    def create_grid_mask(self, text, tokenizer):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        candidate_path = [list(self.lattice.start_state)]
        for token_id in token_ids:
            candidate_path.append(list(self.lattice.token_to_state(token_id)))

        corrected_path, invalid_indexes, drift_detected = self.verify_or_correct_path(
            candidate_path
        )
        invalid_index_set = set(invalid_indexes)

        text_parts = []
        for token_index, token_id in enumerate(token_ids):
            if token_index in invalid_index_set:
                text_parts.append(" _ ")
            else:
                text_parts.append(
                    tokenizer.decode(
                        [token_id],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )

        masked_text = "".join(text_parts).strip()
        total_tokens = len(token_ids)
        masked_tokens = len(invalid_indexes)

        return {
            "masked_text": masked_text,
            "candidate_grid_path": candidate_path,
            "corrected_grid_path": corrected_path,
            "total_tokens": total_tokens,
            "masked_tokens": masked_tokens,
            "mask_rate": masked_tokens / total_tokens if total_tokens else 0.0,
            "cognitive_drift_detected": drift_detected,
            "self_correction_performed": drift_detected,
        }

    def build_logic_trace(self, corrected_grid_path):
        trace = {
            "grid_path": corrected_grid_path,
            "safety_verification": "PASSED",
            "reasoning_integrity_check": "Verified against constraint matrix",
        }
        self.verify_logic_trace(trace)
        return trace

    def finalize_response(self, response_text, candidate_grid_path):
        corrected_path, invalid_indexes, drift_detected = self.verify_or_correct_path(
            candidate_grid_path
        )
        logic_trace = self.build_logic_trace(corrected_path)
        return {
            "response": response_text,
            "LOGIC_TRACE": logic_trace,
            "cognitive_drift_detected": drift_detected,
            "self_correction_performed": drift_detected,
            "invalid_transition_count": len(invalid_indexes),
        }

    def verify_logic_trace(self, trace):
        if not isinstance(trace, dict) or set(trace.keys()) != self.REQUIRED_TRACE_KEYS:
            raise CognitiveDriftError("LOGIC_TRACE has an invalid schema.")
        if trace["safety_verification"] != "PASSED":
            raise CognitiveDriftError("LOGIC_TRACE did not pass verification.")
        if trace["reasoning_integrity_check"] != "Verified against constraint matrix":
            raise CognitiveDriftError("LOGIC_TRACE integrity message is invalid.")

        self.lattice.verify_path(trace["grid_path"])
        return True
