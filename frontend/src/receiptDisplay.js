export const WORKFLOWS = {
  express_send: { label: 'Express Send', category: 'TRANSFER' },
  pay_online: { label: 'Pay Online', category: 'PAYMENT' },
  bank_transfer: { label: 'Bank Transfer', category: 'DEBIT' },
}

export const TRANSACTION_TYPES = {
  PAYMENT: { label: 'Payment', origin_role: 'client', destination_role: 'merchant' },
  TRANSFER: { label: 'Transfer', origin_role: 'client', destination_role: 'client' },
  DEBIT: { label: 'Debit', origin_role: 'client', destination_role: 'merchant' },
  CASH_IN: { label: 'Cash In', origin_role: 'merchant', destination_role: 'client' },
  CASH_OUT: { label: 'Cash Out', origin_role: 'client', destination_role: 'merchant' },
}

export const roleLabel = value => ({ client: 'Client', merchant: 'Merchant' })[value] || value

export const minuteTime = value => typeof value === 'string' ? value.slice(0, 5) : ''

// Format the confirmed local values without converting them through a browser timezone.
export const receiptDate = value => {
  const parts = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value || '')
  return parts ? `${parts[3]}/${parts[2]}/${parts[1]}` : ''
}

export const receiptTime = value => {
  const parts = /^(\d{2}):(\d{2})$/.exec(minuteTime(value))
  if (!parts) return ''
  const hour = Number(parts[1])
  return `${String(hour % 12 || 12).padStart(2, '0')}:${parts[2]} ${hour < 12 ? 'am' : 'pm'}`
}
