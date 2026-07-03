@echo off
echo === Health Check ===
curl http://localhost:8000/health
echo.
echo.

echo === Test 1: vague query ===
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t1_vague.json
echo.
echo.

echo === Test 2: recommend ===
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t2_recommend.json
echo.
echo.

echo === Test 3: refine ===
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t3_refine.json
echo.
echo.

echo === Test 4: compare ===
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t4_compare.json
echo.
echo.

echo === Test 5: confirm end ===
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t5_confirm_end.json
echo.
echo.

echo === Test 6: refuse offtopic ===
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t6_refuse_offtopic.json
echo.
echo.

echo === Test 7: refuse injection ===
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t7_refuse_injection.json
echo.
echo.

echo === Test 8: turn budget ===
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d @test_cases/t8_turn_budget.json
echo.
echo.
