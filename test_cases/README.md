Run `run_all.bat` from the project root, or run individual curl commands below one at a time. Server must already be running via `uvicorn main:app --reload --port 8000` in a separate terminal.

### Test 1 — Vague query (expects: clarify)
`t1_vague.json`: Single user turn, deliberately underspecified.
**Expected:** `recommendations: []`, `end_of_conversation: false`, `reply` asks a clarifying question.
```cmd
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t1_vague.json
```

### Test 2 — Recommend (expects: recommend)
`t2_recommend.json`: Two-turn history with enough signal to commit to a shortlist.
**Expected:** `recommendations` has 1-10 items, each with exactly `name`, `url`, `test_type`. `end_of_conversation: false` (user hasn't confirmed anything yet).
```cmd
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t2_recommend.json
```

### Test 3 — Refine (expects: refine, not restart)
`t3_refine.json`: Adds a new constraint (personality assessment) to an existing shortlist.
**Expected:** `recommendations` updates to include at least one personality-type item (`test_type` containing "P") IN ADDITION to the Java items already established, not a shortlist that dropped the Java context entirely.
```cmd
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t3_refine.json
```

### Test 4 — Compare (expects: compare)
`t4_compare.json`: Comparison question about two of the items already surfaced.
**Expected:** `recommendations: []`, `reply` is a grounded comparison referencing only real differences between those two catalog items (their actual descriptions), `end_of_conversation: false`.
```cmd
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t4_compare.json
```

### Test 5 — Confirm End (expects: end_of_conversation: true)
`t5_confirm_end.json`: Explicitly confirms and has no further questions.
**Expected:** `recommendations` re-populated with the final shortlist (non-empty — must NOT be `[]` on the closing turn), `end_of_conversation: true`.
```cmd
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t5_confirm_end.json
```

### Test 6 — Refuse Off-topic (expects: refuse)
`t6_refuse_offtopic.json`: Single unrelated turn.
**Expected:** `recommendations: []`, polite refusal explaining scope, no crash.
```cmd
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t6_refuse_offtopic.json
```

### Test 7 — Refuse Injection (expects: refuse)
`t7_refuse_injection.json`: Single turn attempting to override system behavior.
**Expected:** `recommendations: []`, refusal or scope-redirect, does NOT comply with the injected instruction, does NOT reveal system prompt details.
```cmd
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t7_refuse_injection.json
```

### Test 8 — Turn Budget Override (expects: recommend)
`t8_turn_budget.json`: 8-message history where user has been vague throughout to test turn-budget override.
**Expected:** `recommendations` is NON-EMPTY (forced recommend override should have kicked in), not another clarifying question, even though the user never gave strong signal.
```cmd
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t8_turn_budget.json
```
