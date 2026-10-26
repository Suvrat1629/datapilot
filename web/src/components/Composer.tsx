import { useEffect, useRef, useState } from "react";

const MAX = 4000;

interface Props {
  disabled: boolean;
  onSend: (text: string) => void;
  focusKey: string;
}

export default function Composer({ disabled, onSend, focusKey }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    ref.current?.focus();
  }, [focusKey, disabled]);

  const submit = () => {
    const value = text.trim();
    if (!value || disabled) return;
    onSend(value);
    setText("");
  };

  return (
    <div className="composer">
      <textarea
        ref={ref}
        value={text}
        rows={1}
        maxLength={MAX}
        placeholder={disabled ? "DataPilot is working…" : "Ask about your data, or ask it to load some…"}
        disabled={disabled}
        onChange={(e) => {
          setText(e.target.value);
          e.target.style.height = "auto";
          e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
      />
      <button className="send" onClick={submit} disabled={disabled || !text.trim()} aria-label="Send">
        ↑
      </button>
      {text.length > MAX - 300 && <span className="count">{MAX - text.length}</span>}
    </div>
  );
}
