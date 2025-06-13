----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 02/04/2025 12:19:48 PM
-- Design Name: 
-- Module Name: SAR_ADC_2bit - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------

library IEEE;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
--use ieee.std_logic_arith.all;
--use ieee.std_logic_unsigned.all;

entity SAR_ADC_2bit is
    port (
        clk : in std_logic; -- System clock
        rst_n : in std_logic; -- Reset (Active Low)
        start : in std_logic; -- Start conversion
        overflow : in std_logic; -- Incoming overflow flag from the comparator
        comparator_result : in std_logic_vector (1 downto 0); -- Incoming comparator result
        sar_out : out std_logic_vector (15 downto 0); -- SAR ADC output
        DAC_minus : out std_logic_vector (15 downto 0); -- Low-side DAC output
        DAC_plus : out std_logic_vector (15 downto 0); -- High-side DAC output
        done_flag : out std_logic -- Conversion done flag 
    );
end entity;

architecture Behavioral of SAR_ADC_2bit is
-- State enumeration
type state_type is (IDLE, UPDATE_DACS, SET_BITS, WAIT_SETTLING, COMPARE, DONE);
signal current_state, next_state : state_type := IDLE;

-- SAR ADC internal signals and registers
signal SAR_reg : std_logic_vector(15 downto 0) := (others => '0');
signal cycle_counter : integer range 0 to 7 := 0; -- Cycle counter for bit setting - * cycles for 16-bit ADC
signal Vmin, Vmax : std_logic_vector(15 downto 0) := (others => '0'); -- Voltage range for the DAC
signal bit_pos : integer range 0 to 15 := 14; -- Current bit position tracker for 2-bit per cycle


-- Settling time for DAC stabilisation
signal settling_timer: integer range 0 to 90 := 0; -- 110 clk cycles (1us @ 100MHz) settling time

begin

-- State transition process
process (clk, rst_n)
begin
    if rst_n = '0' then
        current_state <= IDLE;
    elsif rising_edge(clk) then
        current_state <= next_state;
    end if;
end process;

-- State machine process
process (current_state, start, cycle_counter, overflow, settling_timer)
begin
    case current_state is
        when IDLE =>
            if start = '1' then
                --Vmin <= (others => '0'); -- Initialise Vmin
                --Vmax <= (others => '1'); -- Initialise Vmax
                --bit_pos <= 14; -- Initialise bit position
                next_state <= UPDATE_DACS;
            else
                next_state <= IDLE;
            end if;

        when UPDATE_DACS =>
            next_state <= WAIT_SETTLING;
            
        when WAIT_SETTLING =>
            if settling_timer = 90 then
                next_state <= COMPARE; -- Move to comparison after settling time
            else
                next_state <= WAIT_SETTLING;
            end if;

            when COMPARE =>
                if cycle_counter < 7 then
                    next_state <= UPDATE_DACS; -- Continue to the next SAR cycle
                else
                    next_state <= DONE; -- Finalise conversion
                end if;
            
            when DONE =>
                next_state <= IDLE; -- Return to IDLE for the next operation

            when others =>
                next_state <= IDLE;
    end case;
end process;

-- SAR ADC operation process
process (clk, rst_n)
begin
    if rst_n = '0' then
        sar_reg <= (others => '0');
        cycle_counter <= 0;
        done_flag <= '0';
        settling_timer <= 0;
        Vmin <= (others => '0');
        Vmax <= (others => '1');
    elsif rising_edge(clk) then
        case current_state is
            when idle =>
                cycle_counter <= 0;
                done_flag <= '0';
                settling_timer <= 0;
                bit_pos <= 14;

            when UPDATE_DACS =>
                if cycle_counter > 0 then
                    Vmin <= std_logic_vector(unsigned(SAR_reg) sll (bit_pos+2));
                    Vmax <= std_logic_vector((unsigned(SAR_reg) sll (bit_pos+2)) or ((TO_UNSIGNED(1,16)sll(bit_pos+2))- TO_UNSIGNED(1,16)));
                elsif cycle_counter = 0 then
                    Vmin <= (others => '0');
                    Vmax <= (others => '1');
                end if;
                settling_timer <= 0;
            
            when WAIT_SETTLING =>
                if settling_timer < 110 then
                    settling_timer <= settling_timer + 1; -- Wait for the DAC to settle 
                end if;
            
            when COMPARE =>
                -- Update SAR register using SAR_reg = (SAR_reg << 2) | comparator_result
                SAR_reg <= SAR_reg(13 downto 0) & comparator_result;
                cycle_counter <= cycle_counter + 1;
                bit_pos <= bit_pos - 2; -- Move to the next bits
            
            when DONE =>
                --sar_out <= SAR_reg; -- Store the final result
                done_flag <= '1'; -- Indicate conversion is complete - Assert RDY
                --DAC_plus <= Vmax; -- Set High-side DAC output
                --DAC_minus <= Vmin; -- Set Low-side DAC output
            
            when others =>
                null;
        end case;
    end if;
end process;
DAC_plus <= Vmax; -- Set High-side DAC output
DAC_minus <= Vmin; -- Set Low-side DAC output
SAR_out <= SAR_reg;
end architecture;
