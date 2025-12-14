/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'airport-blue': '#1e40af',
        'airport-cyan': '#06b6d4',
      }
    },
  },
  plugins: [],
}
