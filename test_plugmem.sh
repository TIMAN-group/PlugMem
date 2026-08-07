#!/usr/bin/env bash
# Integration test: Hermes PlugMem memory provider — all features
# Idempotent: creates a unique graph per run. Requires PlugMem service on localhost:8080.
# Usage: bash test_plugmem.sh

set -o pipefail
BASE="http://localhost:8080/api/v1"
GRAPH="test-$(date +%s)"
SHARED="shared-test-$(date +%s)"
PASS=0
FAIL=0

check() {
    local label="$1" expected="$2" actual="$3"
    if echo "$actual" | grep -q "$expected"; then
        echo "  ✅ $label"
        PASS=$((PASS+1))
    else
        echo "  ❌ $label"
        echo "     expected: '$expected'"
        echo "     got:      ${actual:0:200}"
        FAIL=$((FAIL+1))
    fi
}

echo "=== 1. Service health ==="
R=$(curl -s "$BASE/health")
check "health endpoint ok" '"status":"ok"' "$R"

echo ""
echo "=== 2. Graph management ==="
R=$(curl -s -X POST "$BASE/graphs" -H 'Content-Type: application/json' -d "{\"graph_id\":\"$GRAPH\"}")
check "create graph" '"graph_id"' "$R"

R=$(curl -s "$BASE/graphs/$GRAPH/stats")
check "get stats returns" "semantic" "$R"

echo ""
echo "=== 3. Semantic memory (plugmem_remember) ==="
R=$(curl -s -X POST "$BASE/graphs/$GRAPH/memories" -H 'Content-Type: application/json' -d '{
  "mode":"structured",
  "semantic":[
    {"semantic_memory":"Alex prefers dark mode and vim keybindings","tags":["preference","ui"]},
    {"semantic_memory":"Alex hates Entity Framework, prefers Dapper","tags":["preference","tech"]},
    {"semantic_memory":"HyperMesh is a distributed metagraph DB by Alex","tags":["project"]}
  ]
}')
check "insert 3 facts" '"status":"ok"' "$R"
check "semantic count is 3" '"semantic":3' "$R"

echo ""
echo "=== 4. Recall with reasoning ==="
R=$(curl -s -X POST "$BASE/graphs/$GRAPH/reason" -H 'Content-Type: application/json' \
  -d '{"observation":"What are Alex software preferences?"}')
check "reasoning mentions dark mode" "dark mode" "$R"
check "reasoning mentions EF" "Entity Framework" "$R"
check "reasoning mentions HyperMesh" "HyperMesh" "$R"

echo ""
echo "=== 5. Procedural memory (plugmem_learn) ==="
R=$(curl -s -X POST "$BASE/graphs/$GRAPH/memories" -H 'Content-Type: application/json' -d '{
  "mode":"trajectory",
  "goal":"Deploy .NET to Windows VPS",
  "steps":[
    {"observation":"Need to deploy","action":"dotnet publish -c Release"},
    {"observation":"Published","action":"copy to VPS site folder"},
    {"observation":"Uploaded","action":"restart app pool"}
  ]
}')
check "store procedure trajectory" '"status":"ok"' "$R"

R=$(curl -s -X POST "$BASE/graphs/$GRAPH/reason" -H 'Content-Type: application/json' \
  -d '{"observation":"How do I deploy a .NET app?","mode":"procedural_memory"}')
check "recall procedure" "deploy\|publish\|VPS" "$R"

echo ""
echo "=== 6. Consolidation ==="
R=$(curl -s -X POST "$BASE/graphs/$GRAPH/consolidate" -H 'Content-Type: application/json' -d '{}')
check "consolidate succeeds" '"status":"ok"' "$R"

echo ""
echo "=== 7. Shared graph queries ==="
curl -s -X POST "$BASE/graphs" -H 'Content-Type: application/json' -d "{\"graph_id\":\"$SHARED\"}" > /dev/null
curl -s -X POST "$BASE/graphs/$SHARED/memories" -H 'Content-Type: application/json' -d '{
  "mode":"structured",
  "semantic":[{"semantic_memory":"Team uses Caddy as reverse proxy on VPS","tags":["infra","shared"]}]
}' > /dev/null
R=$(curl -s -X POST "$BASE/graphs/$SHARED/reason" -H 'Content-Type: application/json' \
  -d '{"observation":"What reverse proxy does the team use?"}')
check "query shared graph returns Caddy" "Caddy" "$R"

echo ""
echo "=== 8. Episodic trajectory ==="
R=$(curl -s -X POST "$BASE/graphs/$GRAPH/memories" -H 'Content-Type: application/json' -d '{
  "mode":"trajectory",
  "goal":"Debug API 500 error",
  "steps":[
    {"observation":"API returns 500","action":"Checked server logs"},
    {"observation":"NullReferenceException in GetUser","action":"Added null check"},
    {"observation":"Fixed","action":"Deployed and verified"}
  ]
}')
check "store episode trajectory" '"status":"ok"' "$R"

R=$(curl -s -X POST "$BASE/graphs/$GRAPH/reason" -H 'Content-Type: application/json' \
  -d '{"observation":"How was the 500 error resolved?"}')
check "recall episode mentions NullReference" "Null" "$R"

echo ""
echo "==============================================="
echo " Results: $PASS passed, $FAIL failed of $((PASS+FAIL))"
echo "==============================================="
[ $FAIL -eq 0 ] && echo "ALL TESTS PASSED ✅" || echo "SOME TESTS FAILED ❌"
exit $FAIL
