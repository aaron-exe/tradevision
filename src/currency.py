import streamlit as st

# Hardcoded exchange rate – update this single value when the rate changes.
USD_TO_INR = 94.47

def get_currency_symbol():
    """Return the active currency symbol."""
    try:
        return '₹' if st.session_state.get('currency') == 'INR' else '$'
    except Exception:
        return '$'

def get_currency_code():
    """Return the active currency code."""
    try:
        return 'INR' if st.session_state.get('currency') == 'INR' else 'USD'
    except Exception:
        return 'USD'

def convert_currency(amount):
    """Convert a USD amount based on selected currency."""
    try:
        if st.session_state.get('currency') == 'INR':
            return amount * USD_TO_INR
        return amount
    except Exception:
        return amount

def format_currency(amount, fmt=".2f", plus=False):
    """
    Format a monetary amount with the correct currency symbol.

    Args:
        amount: Numeric value in USD.
        fmt: Format spec for the number (default ".2f").
        plus: If True, prepend '+' for positive values.
    Returns:
        Formatted string, e.g. "$123.45" or "₹10,308.08".
    """
    converted = convert_currency(amount)
    sym = get_currency_symbol()
    sign = '+' if plus and converted > 0 else ''
    return f"{sym}{sign}{converted:{fmt}}"

def format_number(num):
    """Format large monetary numbers with currency symbol (T/B/M/K)."""
    converted = convert_currency(num)
    sym = get_currency_symbol()
    if converted >= 1e12:
        return f"{sym}{converted/1e12:.2f}T"
    elif converted >= 1e9:
        return f"{sym}{converted/1e9:.2f}B"
    elif converted >= 1e6:
        return f"{sym}{converted/1e6:.2f}M"
    elif converted >= 1e3:
        return f"{sym}{converted/1e3:.2f}K"
    else:
        return f"{sym}{converted:,.2f}"
