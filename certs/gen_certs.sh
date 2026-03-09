#!/usr/bin/env bash
set -euo pipefail

CERT_DIR="${1:-certs}"
NUM_CLIENTS="${2:-10}"

mkdir -p "$CERT_DIR"
cd "$CERT_DIR"

echo "[+] Generating CA..."
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 \
  -subj "/CN=FL-CA" -out ca.crt

echo "[+] Generating server key/cert..."
openssl genrsa -out server.key 4096
openssl req -new -key server.key -subj "/CN=localhost" -out server.csr

# Add SAN for localhost (important for modern TLS)
cat > server.ext << 'EOF'
subjectAltName = @alt_names
extendedKeyUsage = serverAuth
keyUsage = digitalSignature, keyEncipherment
[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
EOF

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.crt -days 3650 -sha256 -extfile server.ext

echo "[+] Generating client certs (for mTLS)..."
for i in $(seq 0 $((NUM_CLIENTS-1))); do
  CID="client_${i}"
  openssl genrsa -out "${CID}.key" 2048
  openssl req -new -key "${CID}.key" -subj "/CN=${CID}" -out "${CID}.csr"

  cat > "${CID}.ext" << EOF
extendedKeyUsage = clientAuth
keyUsage = digitalSignature, keyEncipherment
EOF

  openssl x509 -req -in "${CID}.csr" -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out "${CID}.crt" -days 3650 -sha256 -extfile "${CID}.ext"
done

echo "[+] Done. Files in: $(pwd)"
echo "    CA:      ca.crt / ca.key"
echo "    Server:  server.crt / server.key"
echo "    Clients: client_0.crt/key ... client_9.crt/key"
