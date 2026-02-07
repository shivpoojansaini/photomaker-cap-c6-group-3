# Multi-identity prompt parser for PhotoMaker
# Parses natural language prompts to extract per-region identity descriptions

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


@dataclass
class RegionPrompt:
    """Represents a parsed region with its associated prompt"""
    region_id: str              # "left", "right", "center", "person1", etc.
    description: str            # Full description with trigger word
    spatial_hint: str           # "left", "right", "center", etc.
    trigger_word: str = "img"   # Position where ID embedding goes


@dataclass
class ParsedMultiPrompt:
    """Complete parsed multi-identity prompt"""
    regions: List[RegionPrompt]
    shared_context: str = ""    # Common context (e.g., "in a park")
    original_prompt: str = ""   # Original input prompt


class MultiIdentityPromptParser:
    """
    Parse natural language prompts to extract per-region identity descriptions.

    Supported patterns:
    - "left person as astronaut, right person as doctor"
    - "the man on the left is a chef, the woman on the right is a scientist"
    - "person1 as astronaut, person2 as doctor"
    """

    # Position keyword mappings
    POSITION_KEYWORDS: Dict[str, List[str]] = {
        'left': ['left', 'on the left', 'leftmost', 'first'],
        'right': ['right', 'on the right', 'rightmost', 'second'],
        'center': ['center', 'middle', 'in the middle', 'central'],
    }

    # Person type keywords
    PERSON_KEYWORDS = ['person', 'man', 'woman', 'guy', 'girl', 'individual', 'one']

    # Connector keywords between position and description
    CONNECTOR_KEYWORDS = ['as', 'is', 'wearing', 'dressed as', 'in', 'as a', 'is a']

    def __init__(self, trigger_word: str = "img"):
        self.trigger_word = trigger_word
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for parsing"""
        # Pattern for explicit position-based descriptions
        # Matches: "left person as astronaut" or "the man on the left is a chef"
        position_group = '|'.join([
            keyword
            for keywords in self.POSITION_KEYWORDS.values()
            for keyword in keywords
        ])
        person_group = '|'.join(self.PERSON_KEYWORDS)
        connector_group = '|'.join(self.CONNECTOR_KEYWORDS)

        # Main pattern: captures position, person type, and description
        self.position_pattern = re.compile(
            rf'(?:the\s+)?'                           # Optional "the"
            rf'({person_group})'                      # Person type (group 1)
            rf'\s+(?:who\s+is\s+)?'                   # Optional "who is"
            rf'(?:on\s+the\s+)?({position_group})'    # Position (group 2)
            rf'\s+(?:{connector_group})\s+'           # Connector
            rf'([^,]+)',                              # Description (group 3)
            re.IGNORECASE
        )

        # Alternative pattern: position first
        self.position_first_pattern = re.compile(
            rf'({position_group})'                    # Position (group 1)
            rf'\s+({person_group})'                   # Person type (group 2)
            rf'\s+(?:{connector_group})\s+'           # Connector
            rf'([^,]+)',                              # Description (group 3)
            re.IGNORECASE
        )

        # Pattern for numbered persons: "person1 as X, person2 as Y"
        self.numbered_pattern = re.compile(
            rf'(person\s*\d+)'                        # Person identifier (group 1)
            rf'\s+(?:{connector_group})\s+'           # Connector
            rf'([^,]+)',                              # Description (group 2)
            re.IGNORECASE
        )

    def parse(self, prompt: str, num_identities: int = 2) -> ParsedMultiPrompt:
        """
        Parse a multi-identity prompt into structured regions.

        Args:
            prompt: Natural language prompt with multiple identity descriptions
            num_identities: Expected number of identities (for validation/fallback)

        Returns:
            ParsedMultiPrompt with regions and shared context
        """
        regions = []

        # Try position-first pattern (e.g., "left person as astronaut")
        matches = self.position_first_pattern.findall(prompt)
        if matches:
            for position, person_type, description in matches:
                spatial_hint = self._normalize_position(position)
                full_desc = self._build_description(person_type, description)
                regions.append(RegionPrompt(
                    region_id=spatial_hint,
                    description=full_desc,
                    spatial_hint=spatial_hint,
                    trigger_word=self.trigger_word
                ))

        # Try alternative pattern (e.g., "the man on the left is a chef")
        if not regions:
            matches = self.position_pattern.findall(prompt)
            if matches:
                for person_type, position, description in matches:
                    spatial_hint = self._normalize_position(position)
                    full_desc = self._build_description(person_type, description)
                    regions.append(RegionPrompt(
                        region_id=spatial_hint,
                        description=full_desc,
                        spatial_hint=spatial_hint,
                        trigger_word=self.trigger_word
                    ))

        # Try numbered pattern (e.g., "person1 as astronaut")
        if not regions:
            matches = self.numbered_pattern.findall(prompt)
            if matches:
                spatial_assignments = ['left', 'right', 'center'][:num_identities]
                for i, (person_id, description) in enumerate(matches[:num_identities]):
                    spatial_hint = spatial_assignments[i] if i < len(spatial_assignments) else f'region{i}'
                    full_desc = self._build_description('person', description)
                    regions.append(RegionPrompt(
                        region_id=person_id.lower().replace(' ', ''),
                        description=full_desc,
                        spatial_hint=spatial_hint,
                        trigger_word=self.trigger_word
                    ))

        # Fallback: split by comma and assign left/right
        if not regions:
            regions = self._parse_comma_separated(prompt, num_identities)

        # Extract shared context
        shared_context = self._extract_shared_context(prompt, regions)

        return ParsedMultiPrompt(
            regions=regions,
            shared_context=shared_context,
            original_prompt=prompt
        )

    def _normalize_position(self, position: str) -> str:
        """Convert position keywords to standard form"""
        pos_lower = position.lower().strip()

        for standard, variants in self.POSITION_KEYWORDS.items():
            if pos_lower in variants or pos_lower == standard:
                return standard

        # Handle numbered positions
        if 'first' in pos_lower:
            return 'left'
        elif 'second' in pos_lower:
            return 'right'

        return pos_lower

    def _build_description(self, person_type: str, description: str) -> str:
        """Build full description with trigger word"""
        description = description.strip().rstrip(',').strip()
        person_type = person_type.strip().lower()

        # Check if trigger word already in description
        if self.trigger_word in description:
            return f"{person_type} {description}"

        # Insert trigger word after person type
        return f"{person_type} {self.trigger_word} {description}"

    def _parse_comma_separated(self, prompt: str, num_identities: int) -> List[RegionPrompt]:
        """Fallback: parse comma-separated descriptions"""
        regions = []
        spatial_assignments = ['left', 'right', 'center'][:num_identities]

        # Split by comma or "and"
        parts = re.split(r',\s*|\s+and\s+', prompt)

        for i, part in enumerate(parts[:num_identities]):
            part = part.strip()
            if not part:
                continue

            spatial_hint = spatial_assignments[i] if i < len(spatial_assignments) else f'region{i}'

            # Ensure trigger word is present
            if self.trigger_word not in part:
                # Try to find person keyword and insert trigger after it
                for person_kw in self.PERSON_KEYWORDS:
                    if person_kw in part.lower():
                        part = re.sub(
                            rf'({person_kw})',
                            rf'\1 {self.trigger_word}',
                            part,
                            count=1,
                            flags=re.IGNORECASE
                        )
                        break
                else:
                    # No person keyword found, prepend "person img"
                    part = f"person {self.trigger_word} {part}"

            regions.append(RegionPrompt(
                region_id=spatial_hint,
                description=part,
                spatial_hint=spatial_hint,
                trigger_word=self.trigger_word
            ))

        return regions

    def _extract_shared_context(self, prompt: str, regions: List[RegionPrompt]) -> str:
        """Extract context shared across all regions"""
        # Look for common context phrases
        context_patterns = [
            r'(?:both\s+)?(?:are\s+)?in\s+(?:a\s+)?(.+?)(?:,|$)',  # "in a park"
            r'(?:at\s+)?(?:the\s+)?(.+?)(?:together|$)',           # "at the beach"
            r'background[:\s]+(.+?)(?:,|$)',                        # "background: city"
        ]

        shared = ""
        for pattern in context_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                shared = match.group(1).strip()
                break

        return shared

    def get_individual_prompts(
        self,
        parsed: ParsedMultiPrompt,
        base_prompt: str = ""
    ) -> List[str]:
        """
        Generate individual prompts for each region.

        Args:
            parsed: Parsed multi-prompt result
            base_prompt: Optional base prompt to prepend

        Returns:
            List of complete prompts, one per identity
        """
        prompts = []
        for region in parsed.regions:
            prompt = region.description
            if base_prompt:
                prompt = f"{base_prompt}, {prompt}"
            if parsed.shared_context:
                prompt = f"{prompt}, {parsed.shared_context}"
            prompts.append(prompt)

        return prompts


# Convenience function
def parse_multi_identity_prompt(
    prompt: str,
    num_identities: int = 2,
    trigger_word: str = "img"
) -> ParsedMultiPrompt:
    """
    Parse a multi-identity prompt.

    Args:
        prompt: Natural language prompt
        num_identities: Expected number of identities
        trigger_word: Trigger word for ID embedding position

    Returns:
        ParsedMultiPrompt with regions and context

    Example:
        >>> result = parse_multi_identity_prompt(
        ...     "left person as astronaut, right person as doctor"
        ... )
        >>> len(result.regions)
        2
        >>> result.regions[0].spatial_hint
        'left'
    """
    parser = MultiIdentityPromptParser(trigger_word)
    return parser.parse(prompt, num_identities)
